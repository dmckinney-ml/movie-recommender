#!/bin/bash

# Load environment variables
source "$(cd $(dirname $BASH_SOURCE) && pwd)/env.sh"

# Check if IMAGE_URI is set
if [ -z "$IMAGE_URI" ]; then
  echo "Error: IMAGE_URI is not set. Please ensure env.sh is configured correctly."
  exit 1
fi

# Run the Docker container interactively
docker run -it --rm \
  --entrypoint=/bin/bash \
  -w /src \
  -v "$COMPONENT_DIR:/src" \
  -v "${HOME}/.config/gcloud:/root/.config/gcloud:ro" \
  $IMAGE_URI
