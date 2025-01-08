# Use a lightweight Python 3.7.3 image as the base
FROM python:3.10-slim-bookworm


# Set the working directory
WORKDIR /src

RUN apt-get update && \
    apt-get install --yes --no-install-recommends

# Copy the application code into the container
COPY . /src

RUN pip install --no-cache-dir -r requirements.txt

# Set the entrypoint to run the main script
ENTRYPOINT ["python3", "/src/main.py"]
