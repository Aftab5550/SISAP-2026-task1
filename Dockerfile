FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libomp-dev \
    libhdf5-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY . .

RUN mkdir -p /app/results

RUN rm -rf build && mkdir build && cd build && cmake .. && make

ENV OMP_NUM_THREADS=8

ENTRYPOINT ["./build/main"]