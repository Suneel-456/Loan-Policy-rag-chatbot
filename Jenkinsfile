// CI/CD pipeline: test -> build images -> push -> (optional) deploy to Kubernetes.
// Needs on the Jenkins agent: python3 (with venv), docker, kubectl.
// Jenkins credentials to create (these are infrastructure credentials, not LLM API keys):
//   dockerhub-credentials : "Username with password"  (your Docker Hub login or access token)
//   kubeconfig            : "Secret file"              (kubeconfig of the target cluster)
pipeline {
    agent any

    options {
        timestamps()
        disableConcurrentBuilds()
        timeout(time: 60, unit: 'MINUTES')
    }

    parameters {
        string(name: 'REGISTRY_USER', defaultValue: 'YOUR_DOCKERHUB_USER', description: 'Docker Hub username (image namespace)')
        booleanParam(name: 'DEPLOY', defaultValue: false, description: 'Deploy to Kubernetes after pushing the images')
    }

    environment {
        API_IMAGE = "${params.REGISTRY_USER}/loan-rag-api"
        UI_IMAGE  = "${params.REGISTRY_USER}/loan-rag-ui"
        IMAGE_TAG = "${env.BUILD_NUMBER}"
        // tests use tiny stand-in models, so nothing is downloaded in CI
        USE_FAKE_MODELS = '1'
    }

    stages {
        stage('Checkout') {
            steps {
                checkout scm
            }
        }

        stage('Test') {
            steps {
                sh '''
                    python3 -m venv .venv
                    . .venv/bin/activate
                    pip install --upgrade pip
                    pip install -r requirements-ci.txt
                    mkdir -p reports
                    python -m pytest -q --junitxml=reports/junit.xml
                '''
            }
            post {
                always {
                    junit allowEmptyResults: true, testResults: 'reports/junit.xml'
                }
            }
        }

        stage('Build images') {
            steps {
                sh '''
                    docker build -t $API_IMAGE:$IMAGE_TAG -t $API_IMAGE:latest -f Dockerfile .
                    docker build -t $UI_IMAGE:$IMAGE_TAG  -t $UI_IMAGE:latest  -f Dockerfile.ui .
                '''
            }
        }

        stage('Push images') {
            steps {
                withCredentials([usernamePassword(credentialsId: 'dockerhub-credentials',
                                                  usernameVariable: 'DOCKER_USER',
                                                  passwordVariable: 'DOCKER_PASS')]) {
                    sh '''
                        echo "$DOCKER_PASS" | docker login -u "$DOCKER_USER" --password-stdin
                        docker push $API_IMAGE:$IMAGE_TAG
                        docker push $API_IMAGE:latest
                        docker push $UI_IMAGE:$IMAGE_TAG
                        docker push $UI_IMAGE:latest
                    '''
                }
            }
        }

        stage('Deploy to Kubernetes') {
            when {
                expression { return params.DEPLOY }
            }
            steps {
                withCredentials([file(credentialsId: 'kubeconfig', variable: 'KUBECONFIG')]) {
                    sh '''
                        kubectl apply -k k8s/
                        kubectl -n loan-rag set image deployment/loan-rag-api api=$API_IMAGE:$IMAGE_TAG
                        kubectl -n loan-rag set image deployment/loan-rag-ui  ui=$UI_IMAGE:$IMAGE_TAG
                        # the API downloads models on first start, so allow a long rollout
                        kubectl -n loan-rag rollout status deployment/loan-rag-api --timeout=1200s
                        kubectl -n loan-rag rollout status deployment/loan-rag-ui  --timeout=300s
                    '''
                }
            }
        }
    }

    post {
        always {
            sh 'docker logout || true'
        }
        success {
            echo "Build ${env.BUILD_NUMBER} finished: ${env.API_IMAGE}:${env.IMAGE_TAG}"
        }
        failure {
            echo 'Pipeline failed. Check the stage logs above.'
        }
    }
}