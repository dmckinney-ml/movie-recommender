# Get the directory where the script resides
SCRIPTS_DIR=$(cd $(dirname $BASH_SOURCE) && pwd)

# Component directory and name
COMPONENT_DIR=$(dirname $SCRIPTS_DIR)
COMPONENT_NAME=$(basename $COMPONENT_DIR)

# Google Cloud project information
PROJECT_ID=$(gcloud config list project --format "value(core.project)")
if [ -z "$PROJECT_ID" ]; then
  echo "Error: No Google Cloud project is set. Use 'gcloud config set project PROJECT_ID' to configure it."
  exit 1
fi

BUCKET_NAME=${BUCKET_NAME:-movie-data-1} # Change this to your bucket name or customize it
# Artifact Registry region and repository name
ARTIFACT_REGISTRY_REGION=${ARTIFACT_REGISTRY_REGION:-us-central1} # Default to us-central1 if not set
REPOSITORY_NAME=${REPOSITORY_NAME:-oolola-repo} # Change this to your repository name or customize it

# Docker image details
IMAGE_NAME=${IMAGE_NAME:-moviedata-$COMPONENT_NAME}
IMAGE_URI=$ARTIFACT_REGISTRY_REGION-docker.pkg.dev/$PROJECT_ID/$REPOSITORY_NAME/$IMAGE_NAME:latest

# Dockerfile location
DOCKERFILE=${DOCKERFILE:-$COMPONENT_DIR/Dockerfile}

# Print environment variables for confirmation
echo "Environment Variables:"
echo "  SCRIPTS_DIR: $SCRIPTS_DIR"
echo "  COMPONENT_DIR: $COMPONENT_DIR"
echo "  COMPONENT_NAME: $COMPONENT_NAME"
echo "  PROJECT_ID: $PROJECT_ID"
echo "  BUCKET_NAME: $BUCKET_NAME"
echo "  ARTIFACT_REGISTRY_REGION: $ARTIFACT_REGISTRY_REGION"
echo "  REPOSITORY_NAME: $REPOSITORY_NAME"
echo "  IMAGE_NAME: $IMAGE_NAME"
echo "  IMAGE_URI: $IMAGE_URI"
echo "  DOCKERFILE: $DOCKERFILE"
