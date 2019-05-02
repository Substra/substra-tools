# substratools

Python package with:
- base class for algo submitted on the Substra Platform (`SubstraModel`)


[Dockerfile](./Dockerfile) is the the dockerfile to build the base image for algo submission. 

# Pull substratools image from private docker registry

- Install Google Cloud SDK: https://cloud.google.com/sdk/install
- Configure docker: `gcloud auth configure-docker`
- Authenticate with registry: `gcloud auth login`
- Pull image: `docker pull eu.gcr.io/substra-208412/substratools`
