FROM nvidia/cudagl:11.3.1-devel-ubuntu20.04
ENV NVIDIA_DRIVER_CAPABILITIES=all
ARG PYTHON_VERSION=3.12.11

# Install os-level packages
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    bash-completion \
    build-essential \
    ca-certificates \
    cmake \
    curl \
    git \
    htop \
    libegl1 \
    libxext6 \
    libjpeg-dev \
    libpng-dev  \
    libvulkan1 \
    rsync \
    tmux \
    unzip \
    vim \
    vulkan-utils \
    wget \
    xvfb \
    && rm -rf /var/lib/apt/lists/*

# Install Miniforge (uses conda-forge, no Terms of Service required)
RUN curl -L -o ~/miniforge.sh https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh && \
    chmod +x ~/miniforge.sh && \
    ~/miniforge.sh -b -p /opt/conda && \
    rm ~/miniforge.sh && \
    /opt/conda/bin/conda init && \
    /opt/conda/bin/conda install -y python="$PYTHON_VERSION" && \
    /opt/conda/bin/conda clean -ya

ENV PATH=/opt/conda/bin:$PATH
SHELL ["/bin/bash", "-c"]

# https://github.com/haosulab/ManiSkill/issues/9
COPY nvidia_icd.json /usr/share/vulkan/icd.d/nvidia_icd.json
COPY nvidia_layers.json /etc/vulkan/implicit_layer.d/nvidia_layers.json

# install dependencies
# RUN pip install --upgrade mani-skill==3.0.0b21 && pip cache purge
# RUN pip install torch