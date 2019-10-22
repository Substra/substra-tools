pipeline {
  options {
    timestamps ()
    timeout(time: 1, unit: 'HOURS')
    buildDiscarder(logRotator(numToKeepStr: '5'))
  }

  agent none

  stages {
    stage('Abort previous builds'){
      steps {
        milestone(Integer.parseInt(env.BUILD_ID)-1)
        milestone(Integer.parseInt(env.BUILD_ID))
      }
    }

    stage('Test & Build') {
      parallel {

        stage('Test') {
          agent {
            kubernetes {
              label 'python'
              defaultContainer 'python'
              yaml """
                apiVersion: v1
                kind: Pod
                spec:
                  containers:
                  - name: python
                    image: python:3.7
                    command: [cat]
                    tty: true
                """
            }
          }

          steps {
            sh "pip install -r requirements.txt"
            sh "pip install flake8"
            sh "flake8"
            sh "pip install -e .[test]"
            sh "python setup.py test"
            sh "pydocmd simple substratools.Algo+ substratools.Metrics+ substratools.Opener+> docs/api.md.tmp  && cmp --silent docs/api.md docs/api.md.tmp"
            sh "rm -rf docs/*.tmp"
          }
        }

        stage('Build') {
          agent {
            kubernetes {
              label 'substratools-kaniko'
              yamlFile '.cicd/agent-kaniko.yaml'
            }
          }

          steps {
            script {
              GIT_DESCRIBE = sh(script: "git describe --always --tags", returnStdout: true).trim()
            }
            withEnv(["GIT_DESCRIBE=${GIT_DESCRIBE}"]) {
              container(name:'kaniko', shell:'/busybox/sh') {
                sh '''#!/busybox/sh
                  /kaniko/executor -f `pwd`/Dockerfile -c `pwd` -d "eu.gcr.io/substra-208412/substra-tools:$GIT_DESCRIBE"
                '''
              }
            }
          }
        }
      }
    }
  }
}
