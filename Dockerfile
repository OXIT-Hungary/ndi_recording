FROM nvidia/cuda:12.6.2-cudnn-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG FFMPEG_VERSION=4.1.11

ENV NVIDIA_DRIVER_CAPABILITIES=all

# Install runtime dependencies
RUN apt-get update -qq --fix-missing && \
    apt-get install -y --no-install-recommends \
        git \
        yasm \
        curl \
        wget \
        avahi-daemon \
        libavahi-client3 \
        dbus \
        libblas-dev \
        libnuma1 \
        libnuma-dev \
        libx264-dev \
        libvpx-dev \
        libass-dev \
        libfreetype6-dev \
        libgnutls28-dev \
        libmp3lame-dev \
        libopus-dev \
        libvorbis-dev \
        python3.10 python3.10-venv python3.10-distutils && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# for docker visualization; todo: check it
RUN apt-get update && apt-get install -y \
    cmake \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    libx11-xcb1 \
    libxcb1 \
    libxcb-render0 \
    libxcb-shape0 \
    libxcb-xfixes0 


# Build FFmpeg with NVIDIA GPU support
WORKDIR /tmp/ffmpeg-build
RUN wget -q https://github.com/FFmpeg/nv-codec-headers/releases/download/n11.0.10.3/nv-codec-headers-11.0.10.3.tar.gz && \
    tar -xzf nv-codec-headers-11.0.10.3.tar.gz && \
    cd nv-codec-headers-11.0.10.3 && \
    make install && \
    cd .. && rm -rf nv-codec-headers-11.0.10.3*

RUN wget -q https://ffmpeg.org/releases/ffmpeg-${FFMPEG_VERSION}.tar.gz && \
    tar -xzf ffmpeg-${FFMPEG_VERSION}.tar.gz && \
    cd ffmpeg-${FFMPEG_VERSION} && \
    ./configure \
        --enable-nonfree \
        --enable-cuda \
        --enable-nvenc \
        --enable-cuvid \
        --enable-libnpp \
        --prefix=/usr/local \
        --pkg-config-flags="--static" \
        --extra-cflags=-I/usr/local/cuda/include \
        --extra-ldflags=-L/usr/local/cuda/lib64 \
        --extra-libs="-lpthread -lm" \
        --bindir="/usr/local/bin" \
        --disable-static \
        --enable-shared && \
    make -j$(nproc) && make install && \
    cd .. && rm -rf ffmpeg-${FFMPEG_VERSION}*

# Set Python 3.10 as the default python version
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 && \
    update-alternatives --set python3 /usr/bin/python3.10

# Copy application code
WORKDIR /app
COPY requirements/prod.txt /app/
# COPY main.py /app/
# COPY app /app/app
# COPY models/rtdetrv2_pano.onnx /app/models/
# COPY src /app/src
# COPY configs /app/configs

# Install Python dependencies in a virtual environment
RUN python3 -m venv ~/.venv/app && \
    ~/.venv/app/bin/pip install --no-cache-dir -r prod.txt

# CMD ["/bin/bash", "-c", "source ~/.venv/app/bin/activate && exec bash"]
CMD ["/bin/bash", "-c", "source ~/.venv/app/bin/activate && uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload"]
