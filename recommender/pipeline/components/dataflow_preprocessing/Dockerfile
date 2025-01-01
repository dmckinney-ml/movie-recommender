# Use a lightweight Python 3.12 image as the base
FROM apache/beam_python3.12_sdk:2.61.0

# Set the working directory
WORKDIR /src

RUN apt-get update && \
    apt-get install --yes --no-install-recommends

# Copy the application code into the container
COPY . /src

# Set the entrypoint to run the main script
ENTRYPOINT ["python3", "/src/main.py"]
