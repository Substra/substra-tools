# substratools

Python package with:
- base class for algo submitted on the Substra Platform (`SubstraModel`)


[Dockerfile](./Dockerfile) is the the dockerfile to build the base image for algo submission. 

# Pull substratools image from private docker registry

- Install Google Cloud SDK: https://cloud.google.com/sdk/install
- Authenticate with your google account: `gcloud auth login`
- Configure docker to use your google credentials for google based docker registery: `gcloud auth configure-docker`
- Pull image: `docker pull eu.gcr.io/substra-208412/substratools`
