# ===============================
# Base image
# ===============================
FROM ubuntu:22.04

# ===============================
# Set environment variables
# ===============================
ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
ENV PATH=/opt/conda/bin:$PATH

# ===============================
# Install essentials
# ===============================
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget git curl bzip2 build-essential \
    ca-certificates libglib2.0-0 libxext6 libsm6 libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# ===============================
# Install Micromamba
# ===============================
RUN wget -qO- https://micromamba.snakepit.net/api/micromamba/linux-64/latest | tar -xvj -C /usr/local/bin --strip-components=1 bin/micromamba

# Create base directories
WORKDIR /app

# ===============================
# Copy environment YAMLs
# ===============================
COPY bacthecom_env.yml ./bacthecom_env.yml

# ===============================
# Create unified environment
# ===============================
RUN micromamba create -y -n bacthecom python=3.10 \
    && micromamba install -y -n bacthecom -f ./bacthecom_env.yml \
    && micromamba clean -a -y

# Activate environment by default
SHELL ["micromamba", "run", "-n", "bacthecom", "/bin/bash", "-c"]

# ===============================
# Copy project code
# ===============================
COPY . /app

# ===============================
# Set default working dir and entrypoint
# ===============================
WORKDIR /app
ENTRYPOINT ["micromamba", "run", "-n", "bacthecom", "python3"]
