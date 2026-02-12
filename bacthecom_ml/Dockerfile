# ===============================
# Base image
# ===============================
FROM ubuntu:22.04 AS runtime

# ===============================
# Set environment variables
# ===============================
ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
ENV MAMBA_ROOT_PREFIX=/opt/conda
ENV PATH=/opt/conda/envs/bacthecom/bin:$PATH

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
RUN wget -qO- https://micromamba.snakepit.net/api/micromamba/linux-64/latest \
    | tar -xvj -C /usr/local/bin --strip-components=1 bin/micromamba

WORKDIR /app

# ===============================
# Copy environment YAML
# ===============================
COPY bacthecom_env.yml ./bacthecom_env.yml

# ===============================
# Create environment
# ===============================
RUN micromamba create -y -n bacthecom python=3.10 \
    && micromamba install -y -n bacthecom -f ./bacthecom_env.yml \
    && micromamba clean -a -y \
    && rm -rf /opt/conda/envs/bacthecom/lib/python*/site-packages/*/tests \
    && rm -rf /opt/conda/envs/bacthecom/lib/python*/site-packages/*/test \
    && ln -s /opt/conda/envs/bacthecom/bin/python /usr/local/bin/python \
    && ln -s /opt/conda/envs/bacthecom/bin/python /usr/local/bin/python3

# ===============================
# Copy project code
# ===============================
COPY . /app

WORKDIR /app