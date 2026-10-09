"""Optional tracing to a self-hosted Langfuse (open source LangSmith alternative).

Off by default. It turns on only if LANGFUSE_HOST, LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY
are set AND the `langfuse` package is installed (pip install -r requirements-observability.txt).
The keys are created inside YOUR OWN Langfuse server, they are not a third-party API key.
If Langfuse is unreachable the app keeps working: tracing errors are swallowed.
The built-in SQLite tracker (src/usage.py) works with or without Langfuse.
Only masked text is ever sent.
"""
import contextlib
import logging

from src import config

log = logging.getLogger(__name__)


class _NullObs:
    def update(self, **_kwargs):
        pass


class NullTracer:
    enabled = False

    @contextlib.contextmanager
    def observe(self, name: str, as_type: str = "span", **kwargs):
        yield _NullObs()

    def shutdown(self):
        pass


class LangfuseTracer:
    enabled = True

    def __init__(self, client):
        self._client = client

    @contextlib.contextmanager
    def observe(self, name: str, as_type: str = "span", **kwargs):
        try:
            cm = self._client.start_as_current_observation(name=name, as_type=as_type, **kwargs)
            obs = cm.__enter__()
        except Exception as exc:  # tracing must never break answering
            log.warning("Langfuse span failed: %s", exc)
            yield _NullObs()
            return
        try:
            yield _SafeObs(obs)
        finally:
            try:
                cm.__exit__(None, None, None)
            except Exception as exc:
                log.warning("Langfuse span close failed: %s", exc)

    def shutdown(self):
        try:
            self._client.shutdown()
        except Exception:
            pass


class _SafeObs:
    def __init__(self, obs):
        self._obs = obs

    def update(self, **kwargs):
        try:
            self._obs.update(**kwargs)
        except Exception as exc:
            log.warning("Langfuse update failed: %s", exc)


def get_tracer():
    if not (config.LANGFUSE_HOST and config.LANGFUSE_PUBLIC_KEY and config.LANGFUSE_SECRET_KEY):
        return NullTracer()
    try:
        from langfuse import Langfuse

        client = Langfuse(
            public_key=config.LANGFUSE_PUBLIC_KEY,
            secret_key=config.LANGFUSE_SECRET_KEY,
            host=config.LANGFUSE_HOST,
        )
        log.info("Langfuse tracing enabled (%s)", config.LANGFUSE_HOST)
        return LangfuseTracer(client)
    except Exception as exc:
        log.warning("Langfuse not available (%s). Continuing without tracing.", exc)
        return NullTracer()