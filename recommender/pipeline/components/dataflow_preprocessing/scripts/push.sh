#!/bin/bash

# Load environment variables
. $(cd $(dirname $BASH_SOURCE) && pwd)/env.sh

# Validate IMAGE_URI
if [ -z "$IMAGE_URI" ]; then
  echo "Error: IMAGE_URI is not set. Please ensure env.sh is configured correctly."
  exit 1
fi

# Authenticate Docker with Artifact Registry
ARTIFACT_REGISTRY_DOMAIN=$(echo $IMAGE_URI | awk -F '/' '{print $1}')
if [[ $ARTIFACT_REGISTRY_DOMAIN == *"docker.pkg.dev"* ]]; then
  echo "Authenticating with Artifact Registry: $ARTIFACT_REGISTRY_DOMAIN"
  gcloud auth configure-docker $ARTIFACT_REGISTRY_DOMAIN
fi

# Push the Docker image
echo "Pushing Docker image to: $IMAGE_URI"
docker push $IMAGE_URI

if [ $? -eq 0 ]; then
  echo "Docker image successfully pushed to: $IMAGE_URI"
else
  echo "Error: Failed to push the Docker image."
  exit 1
fi
