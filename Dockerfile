FROM python:3.11-slim

# Install small system tools some Python packages expect
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl unzip && \
    rm -rf /var/lib/apt/lists/*

# Set the working directory inside the container
WORKDIR /app

# Copy dependency list first (enables Docker layer caching)
COPY requirements.txt .
# Install pinned Python deps
RUN pip install --no-cache-dir -r requirements.txt

# Now copy the rest of your source code
COPY . .

# Ensure the pipeline runner is executable
RUN chmod +x pipeline/run_pipeline.bash

# Default command if you run the image with no args
CMD ["bash"]
