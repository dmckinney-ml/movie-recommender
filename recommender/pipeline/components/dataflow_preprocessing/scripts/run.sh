#!/bin/bash

# Load environment variables from the env.sh file
. $(cd $(dirname $BASH_SOURCE) && pwd)/env.sh

# Ensure that the Docker image URI is available
if [[ -z "$IMAGE_URI" ]]; then
  echo "ERROR: IMAGE_URI is not set. Please ensure that the environment is correctly set up."
  exit 1
fi

# Check if the gcloud configuration directory exists and mount it to Docker
if [ ! -d "${HOME}/.config/gcloud" ]; then
  echo "ERROR: Google Cloud SDK configuration not found at ${HOME}/.config/gcloud."
  echo "Please ensure that gcloud is installed and authenticated."
  exit 1
fi

# Run the Docker container
# Mount Google Cloud SDK config
docker run \
  --platform linux/amd64 \
  -v "${HOME}/.config/gcloud:/root/.config/gcloud" \
  -e GOOGLE_APPLICATION_CREDENTIALS=/root/.config/gcloud/application_default_credentials.json \
  --entrypoint python3 \
  "$IMAGE_URI" \
  /src/main.py

