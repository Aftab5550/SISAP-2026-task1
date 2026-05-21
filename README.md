# Apex SIMD Engine: SISAP 2026 Indexing Challenge (Task 1)

[![Build Status](https://github.com/Aftab5550/SISAP-2026-task1/actions/workflows/main.yml/badge.svg)](https://github.com/Aftab5550/SISAP-2026-task1/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![C++20](https://img.shields.io/badge/C++-20-blue.svg)](https://isocpp.org/)

A high-performance Approximate Nearest Neighbor (ANN) search engine developed for Task 1 of the SISAP 2026 Indexing Challenge. This repository contains a highly optimized C++ implementation designed to compute a k-nearest neighbor graph over massive high-dimensional datasets under strict hardware constraints.

**Author:** Aftab Ahmed Choudhry  
**Institution:** Master in Innovation and Research in Informatics (MIRI), Advanced Data Structures

---

## Performance Benchmarks

The engine was evaluated against the competition's 6.35 million vector dataset (WIKIPEDIA, 1024D) within the official Docker resource limits (8 vCPUs, 24 GB RAM, No Swap).

| Metric | SISAP Baseline Requirement | Apex SIMD Engine Results |
| :--- | :--- | :--- |
| **Recall@15** | > 80.00% | **85.33%** |
| **Execution Time** | < 8 Hours | **~36 Minutes** |
| **Peak RAM (RSS)** | < 24.0 GB | **18.4 GB** |

*Note: The engine severely undercuts the maximum execution time while comfortably exceeding the recall threshold.*

---

## Core Architecture & Systems Engineering

The solution employs a hybrid **Random Projection (RP) Forest** initialization followed by a hardware-accelerated **Nearest Neighbor Descent (NN-Descent)** refinement phase.

To achieve maximum throughput on modern CPUs, the following systems-level optimizations were implemented:
* **Dual-Representation Data Ingestion:** The 25GB HDF5 `float32` dataset is loaded in highly optimized 2.2-million vector chunks. It is simultaneously quantized to `int8_t` and hashed into 128-bit binary sketches.
* **Hardware Popcount Filtering:** During NN-Descent, neighborhood candidates are preemptively filtered using their 128-bit sketches via the hardware `__builtin_popcountll` instruction, avoiding expensive full-dimensional distance calculations.
* **AVX SIMD Vectorization:** By aggressively overriding compiler aliasing constraints (`__restrict`) and utilizing OpenMP SIMD directives, the dot-product computations are forced into AVX registers, processing 64 dimensions simultaneously.
* **Dynamic OpenMP Load Balancing:** Thread starvation caused by varying local neighborhood densities is resolved via `schedule(dynamic, 128)`, keeping all 8 CPU cores fully saturated.
* **Memory Poisoning Sentinels:** The underlying graph data structures utilize preemptive `numeric_limits` boundaries to ensure mathematical safety during lock-free neighborhood updates.

---

## Repository Structure

```text
.
├── src/
│   └── main.cpp                  # The core production C++ engine
├── scripts/
│   ├── data_prep/                # Python scripts for PCA and data exploration
│   ├── evaluation/               # Independent scripts to benchmark .h5 outputs
│   └── plotting/                 # Visualization tools for hyperparameter tuning
├── docs/
│   └── assets/                   # Pareto frontiers, grid search CSVs, and charts
├── archive/                      # Historical R&D iterations (Bitwise, Hybrid approaches)
├── .github/workflows/            # CI/CD Pipeline proving Docker build reproducibility
├── CMakeLists.txt                # C++20 Build configuration
├── Dockerfile                    # SISAP Evaluation Environment spec
└── README.md                     # This file
```

## Reproducibility & Evaluation (Docker)

This repository is fully compliant with the SISAP 2026 blind evaluation framework. The algorithm is containerized and expects the dataset to be mounted as read-only.

### 1. Build the Container
```bash
docker build -t sisap-baseline .
```

### 2. Run the Evaluation
Execute the container using the exact limits specified by the SISAP organizers:
```bash
docker run \
    -it \
    --cpus=8 \
    --memory=24g \
    --memory-swap=24g \
    --memory-swappiness 0 \
    --volume $(pwd)/data:/app/data:ro \
    --volume $(pwd)/results:/app/results:rw \
    sisap-baseline --task task1 --dataset data/benchmark-dev-wikipedia-bge-m3.h5
```
*The engine will automatically output the finalized `knns` and `dists` matrices, alongside the required metadata attributes, into `results/task1/apex_simd.h5`.*

---

## Local Compilation (Development)

To compile and run the engine locally outside of Docker:

**Dependencies:**
* C++20 Compiler (GCC/Clang)
* CMake >= 3.10
* OpenMP
* HDF5 C++ Libraries (`libhdf5-dev`)

**Build Steps:**
```bash
mkdir build && cd build
cmake ..
make
./main --task task1 --dataset ../data/your_dataset.h5
```

---

## License
This project is licensed under the [MIT License](LICENSE) - see the LICENSE file for details.