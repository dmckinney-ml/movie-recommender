#!/bin/bash

# Load environment variables
. $(cd $(dirname $BASH_SOURCE) && pwd)/env.sh

# Authenticate Docker with Artifact Registry
gcloud auth configure-docker $ARTIFACT_REGISTRY_REGION-docker.pkg.dev

# Build the Docker image
docker build $COMPONENT_DIR -f $DOCKERFILE -t $IMAGE_URI

# Push the image to Artifact Registry
docker push $IMAGE_URI

echo "Docker image built and pushed to Artifact Registry: $IMAGE_URI"
