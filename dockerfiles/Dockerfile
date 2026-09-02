FROM nvidia/cuda:13.3.1-runtime-ubuntu24.04 AS cuda-runtime

# The host machine still provides the actual NVIDIA kernel driver. This image installs
# the CUDA runtime libraries needed for GPU-accelerated inference within the container.

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    cuda-nsight-systems-13-3 \
    cuda-cudart-dev-13-3 \
    cuda-libraries-dev-13-3 \
    cuda-nvml-dev-13-3 \
    libcublas-dev-13-3 \
    nvidia-driver-610 \
    nvidia-utils-610 && \
    rm -rf /var/lib/apt/lists/*

ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    PATH=/usr/local/cuda/bin:${PATH} \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/lib:${LD_LIBRARY_PATH}

# Prepare build and runtime environment
RUN apt-get update && \
    DEBIAN_FRONTEND="noninteractive" apt-get install -y --no-install-recommends \
    apt-utils \
    build-essential \
    cmake \
    ccache \
    clang \
    clangd \
    curl \
    dkms \
    flake8 \
    g++ \
    gcc \
    git \
    libsqlite3-dev \
    libssl-dev \
    libclang-dev \
    libgs-dev \
    libopencv-dev \
    libpoppler-cpp-dev \
    llvm \
    llvm-dev \
    nlohmann-json3-dev \
    ninja-build \
    pkg-config \
    software-properties-common \
    sudo \
    tar \
    tzdata \
    unzip \
    vim \
    wget \
    zip && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy configuration bootstrap files
RUN mkdir -p /root/.config/ungulate/bootstrap/config

COPY ./db.json /root/.config/ungulate/bootstrap/config/db.json
COPY ./model.json /root/.config/ungulate/bootstrap/config/model.json
COPY ./net.json /root/.config/ungulate/bootstrap/config/net.json

ARG llama_tag=b9382
ARG nproc=4
ARG username=sam

# Copy compatible ggml libraries
COPY ./llama.cpp.${llama_tag}.tar.xz /llama.cpp.${llama_tag}.tar.xz
RUN tar -xzvf /llama.cpp.${llama_tag}.tar.xz -C / && \
    cd /llama.cpp.${llama_tag}/build && \
    cmake --install . && \
    rm -f ./llama.cpp.${llama_tag}.tar.xz && \
    rm -rf ./llama.cpp.${llama_tag}

COPY ./minja.tar.xz /minja.tar.xz
RUN tar -xzvf /minja.tar.xz -C / && \
    cp -a /minja /usr/local/include/ && \
    rm -rf /minja

# Pull and install ungulate
RUN git clone --recursive https://github.com/sam-ysf/ungulate && \
    mkdir /ungulate/build && cd /ungulate/build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release && \
    make -j ${nproc} && make install && \
    cd / && \
    rm -rf /ungulate

EXPOSE 6767/tcp
EXPOSE 6868/tcp

COPY ./bootstrap.sh /root/.config/ungulate/bootstrap/bootstrap.sh

ENTRYPOINT [ "/bin/bash", "/root/.config/ungulate/bootstrap/bootstrap.sh" ]
