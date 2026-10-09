"""PII masking. Runs on the user's question BEFORE retrieval, the LLM, the cache, the usage
log and any tracing, and again (regex only) on the answer.

Two layers:
  1. Regex rules for Indian identifiers: PAN, Aadhaar, mobile, IFSC, e-mail, card numbers and
     account numbers that follow the word "account". Fast, exact formats, no model.
  2. Microsoft Presidio (open source: presidio-analyzer finds PII, presidio-anonymizer replaces it).
     Adds what regex cannot: person names and places. Needs a spaCy model (en_core_web_sm).
     Dates and organisations are deliberately NOT masked, because loan answers are full of
     durations like "12 months" and "30 days".
Presidio is optional: with PII_ENGINE=auto the app falls back to regex if it is missing.
Masking is a safety net, not a guarantee.
"""
import logging
import re
import threading

from src import config

log = logging.getLogger(__name__)

# Order matters: longer / more specific patterns first.
_PATTERNS: list[tuple[str, re.Pattern, str]] = [
    ("EMAIL", re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"), "[EMAIL]"),
    ("PAN", re.compile(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b"), "[PAN]"),
    ("IFSC", re.compile(r"\b[A-Z]{4}0[A-Z0-9]{6}\b"), "[IFSC]"),
    ("CARD", re.compile(r"(?<!\d)(?:\d[ -]?){12}\d(?:[ -]?\d){0,3}(?!\d)"), "[CARD]"),
    ("AADHAAR", re.compile(r"(?<!\d)\d{4}[ -]?\d{4}[ -]?\d{4}(?!\d)"), "[AADHAAR]"),
    ("PHONE", re.compile(r"(?<!\d)(?:\+?91[\s-]?)?[6-9]\d{4}[\s-]?\d{5}(?!\d)"), "[PHONE]"),
]
_ACCOUNT = re.compile(r"(?i)(account\s*(?:no\.?|number|num)?\s*[:#-]?\s*)(\d{9,18})")

# Presidio entity -> (our label, replacement token)
_PRESIDIO_ENTITIES = {
    "PERSON": ("PERSON", "[PERSON]"),
    "LOCATION": ("LOCATION", "[LOCATION]"),
    "EMAIL_ADDRESS": ("EMAIL", "[EMAIL]"),
    "PHONE_NUMBER": ("PHONE", "[PHONE]"),
    "CREDIT_CARD": ("CARD", "[CARD]"),
    "IP_ADDRESS": ("IP", "[IP]"),
}

_presidio = None          # (analyzer, anonymizer, OperatorConfig) once loaded
_presidio_error = None    # why Presidio is unavailable
_presidio_loaded = False
_load_lock = threading.Lock()


def _load_presidio():
    """Load Presidio once. Returns the engines, or None when unavailable."""
    global _presidio, _presidio_error, _presidio_loaded
    with _load_lock:
        if _presidio_loaded:
            return _presidio
        _presidio_loaded = True
        try:
            import spacy

            # Presidio would try to download a missing model and then call sys.exit();
            # check first so a missing model becomes a normal "unavailable" instead.
            if not spacy.util.is_package(config.PRESIDIO_SPACY_MODEL):
                raise RuntimeError(
                    f"spaCy model '{config.PRESIDIO_SPACY_MODEL}' is not installed "
                    f"(run: python -m spacy download {config.PRESIDIO_SPACY_MODEL})"
                )
            from presidio_analyzer import AnalyzerEngine, RecognizerRegistry
            from presidio_analyzer.nlp_engine import NlpEngineProvider
            from presidio_analyzer.predefined_recognizers import PhoneRecognizer
            from presidio_anonymizer import AnonymizerEngine
            from presidio_anonymizer.entities import OperatorConfig

            nlp = NlpEngineProvider(
                nlp_configuration={
                    "nlp_engine_name": "spacy",
                    "models": [{"lang_code": "en", "model_name": config.PRESIDIO_SPACY_MODEL}],
                }
            ).create_engine()
            registry = RecognizerRegistry()
            registry.load_predefined_recognizers(nlp_engine=nlp)
            registry.remove_recognizer("PhoneRecognizer")
            registry.add_recognizer(PhoneRecognizer(supported_regions=("IN",)))
            analyzer = AnalyzerEngine(nlp_engine=nlp, registry=registry, supported_languages=["en"])
            _presidio = (analyzer, AnonymizerEngine(), OperatorConfig)
        except (Exception, SystemExit) as exc:  # missing package or missing spaCy model
            _presidio_error = f"{type(exc).__name__}: {exc}"
            _presidio = None
        return _presidio


def warm_up() -> str:
    """Load Presidio at startup (so the first question is not slow). Returns the active engine."""
    if config.PII_ENGINE != "regex":
        if _load_presidio() is None:
            if config.PII_ENGINE == "presidio":
                raise RuntimeError(
                    "PII_ENGINE=presidio but Presidio is not available "
                    f"({_presidio_error}). Install: python -m pip install presidio-analyzer "
                    f"presidio-anonymizer spacy && python -m spacy download {config.PRESIDIO_SPACY_MODEL}"
                )
            log.warning("Presidio not available (%s). Using regex PII masking only.", _presidio_error)
    return engine_name()


def engine_name() -> str:
    if config.PII_ENGINE != "regex" and _presidio_loaded and _presidio is not None:
        return "regex+presidio"
    return "regex"


def _regex_mask(text: str) -> tuple[str, dict[str, int]]:
    found: dict[str, int] = {}

    def _account(m: re.Match) -> str:
        found["ACCOUNT"] = found.get("ACCOUNT", 0) + 1
        return m.group(1) + "[ACCOUNT]"

    masked = _ACCOUNT.sub(_account, text)
    for name, pattern, token in _PATTERNS:
        masked, n = pattern.subn(token, masked)
        if n:
            found[name] = found.get(name, 0) + n
    return masked, found


def _presidio_mask(text: str, engines) -> tuple[str, dict[str, int]]:
    analyzer, anonymizer, OperatorConfig = engines
    results = analyzer.analyze(
        text=text,
        language="en",
        entities=list(_PRESIDIO_ENTITIES),
        score_threshold=0.5,
        allow_list=config.PII_ALLOW_LIST,
    )
    if not results:
        return text, {}
    operators = {
        entity: OperatorConfig("replace", {"new_value": token})
        for entity, (_label, token) in _PRESIDIO_ENTITIES.items()
    }
    masked = anonymizer.anonymize(text=text, analyzer_results=results, operators=operators).text
    found: dict[str, int] = {}
    for r in results:
        label = _PRESIDIO_ENTITIES[r.entity_type][0]
        found[label] = found.get(label, 0) + 1
    return masked, found


def mask_pii(text: str, use_presidio: bool = None) -> tuple[str, dict[str, int]]:
    """Return (masked_text, {pii_type: count}).

    use_presidio=None follows PII_ENGINE. The answer-side call passes False: the policy text
    contains no personal names, and Presidio can mistake headings like "Loan Amount" for people.
    """
    masked, found = _regex_mask(text)
    if use_presidio is None:
        use_presidio = config.PII_ENGINE != "regex"
    if use_presidio:
        engines = _load_presidio()
        if engines is None:
            if config.PII_ENGINE == "presidio":
                raise RuntimeError(f"Presidio is not available: {_presidio_error}")
        else:
            masked, extra = _presidio_mask(masked, engines)
            for k, v in extra.items():
                found[k] = found.get(k, 0) + v
    return masked, found