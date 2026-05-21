#include <iostream>
#include <vector>
#include <numeric>
#include <algorithm>
#include <chrono>
#include <fstream>
#include <string>
#include <cmath>
#include <cstdint>
#include <omp.h>
#include <H5Cpp.h>

using namespace std;
using namespace std::chrono;
using namespace H5;

const int NUM_VECTORS = 200000;
const int K = 15;             
const int GT_K = 32;          
const int TEST_QUERIES = 10000;

double get_peak_memory_mb() {
    ifstream file("/proc/self/status");
    string line;
    while (getline(file, line)) {
        if (line.rfind("VmPeak:", 0) == 0) {
            string val = line.substr(7);
            val.erase(remove_if(val.begin(), val.end(), ::isspace), val.end());
            size_t kb_pos = val.find("kB");
            if (kb_pos != string::npos) val = val.substr(0, kb_pos);
            return stod(val) / 1024.0;
        }
    }
    return -1.0;
}

struct Neighbor {
    int id;
    float score;
    // Sorts highest score first
    bool operator<(const Neighbor& other) const { return score > other.score; }
};

int main(int argc, char** argv) {
    auto global_start = steady_clock::now(); 

    if (argc != 4) {
        cerr << "Usage: ./baseline_test <h5_file> <dimensions> <mode>" << endl;
        cerr << "Modes: fp32 | int8 | binary" << endl;
        return -1;
    }

    string filename = argv[1];
    int dim = stoi(argv[2]);
    string mode = argv[3];

    if (mode != "fp32" && mode != "int8" && mode != "binary") {
        cerr << "Error: Invalid mode." << endl;
        return -1;
    }

    cout << "==================================================" << endl;
    cout << "   SISAP 2026: UNIFIED BASELINE ENGINE            " << endl;
    cout << "   File: " << filename << endl;
    cout << "   Dimensions: " << dim << " | Mode: " << mode << endl;
    cout << "==================================================" << endl;

    vector<int> gt_data(NUM_VECTORS * GT_K);
    vector<float> float_data;
    vector<int8_t> int8_data;
    vector<uint64_t> binary_data;

    int num_blocks = dim / 64; // For 1024D, this is 16 blocks of 64 bits

    // 1. I/O TIMING
    auto io_start = steady_clock::now();
    cout << "[1] Loading HDF5 data into RAM..." << endl;
    try {
        H5File file(filename, H5F_ACC_RDONLY);
        float_data.resize(NUM_VECTORS * dim);
        file.openDataSet("/train").read(float_data.data(), PredType::NATIVE_FLOAT);
        file.openDataSet("/allknn/knns").read(gt_data.data(), PredType::NATIVE_INT);
    } catch (const Exception &error) {
        cerr << "Error reading HDF5 file!" << endl;
        return -1;
    }
    auto io_end = steady_clock::now();
    duration<double> io_time = io_end - io_start;

    duration<double> quant_time(0);

    // 2. PREPROCESSING
    if (mode == "int8") {
        auto quant_start = steady_clock::now();
        cout << "[2] Applying Int8 Quantization..." << endl;
        int8_data.resize(NUM_VECTORS * dim);
        #pragma omp parallel for
        for (size_t i = 0; i < float_data.size(); ++i) {
            float scaled = float_data[i] * 127.0f;
            if (scaled > 127.0f) scaled = 127.0f;
            if (scaled < -128.0f) scaled = -128.0f;
            int8_data[i] = static_cast<int8_t>(round(scaled));
        }
        float_data.clear(); float_data.shrink_to_fit();
        quant_time = steady_clock::now() - quant_start;

    } else if (mode == "binary") {
        auto quant_start = steady_clock::now();
        cout << "[2] Applying Binary (1-bit) Quantization..." << endl;
        binary_data.resize(NUM_VECTORS * num_blocks, 0); // Initialize with 0s
        
        #pragma omp parallel for
        for (int i = 0; i < NUM_VECTORS; ++i) {
            for (int d = 0; d < dim; ++d) {
                if (float_data[i * dim + d] > 0.0f) {
                    int block = d / 64;
                    int bit = d % 64;
                    // Bitwise OR to flip the specific bit to 1
                    binary_data[i * num_blocks + block] |= (1ULL << bit);
                }
            }
        }
        float_data.clear(); float_data.shrink_to_fit();
        quant_time = steady_clock::now() - quant_start;
    } else {
        cout << "[2] Mode is fp32. Skipping quantization." << endl;
    }

    // 3. ALGORITHM TIMING
    cout << "[3] Starting Brute Force Search (" << TEST_QUERIES << " queries)..." << endl;
    auto algo_start = steady_clock::now();
    float total_recall = 0.0f;

    #pragma omp parallel for reduction(+:total_recall)
    for (int q = 0; q < TEST_QUERIES; ++q) {
        vector<Neighbor> neighbors(NUM_VECTORS);

        for (int i = 0; i < NUM_VECTORS; ++i) {
            if (q == i) {
                neighbors[i] = {i, -999999.0f}; 
                continue;
            }
            
            if (mode == "int8") {
                int32_t dot_product = 0;
                for (int d = 0; d < dim; ++d) {
                    dot_product += int8_data[q * dim + d] * int8_data[i * dim + d];
                }
                neighbors[i] = {i, static_cast<float>(dot_product)};
            } 
            else if (mode == "binary") {
                int distance = 0;
                for (int d = 0; d < num_blocks; ++d) {
                    // Hamming distance: count bits that are different (XOR)
                    distance += __builtin_popcountll(binary_data[q * num_blocks + d] ^ binary_data[i * num_blocks + d]);
                }
                // Hamming distance is lower-is-better. We use negative so sorting highest-first works properly.
                neighbors[i] = {i, static_cast<float>(-distance)};
            }
            else {
                float dot_product = 0.0f;
                for (int d = 0; d < dim; ++d) {
                    dot_product += float_data[q * dim + d] * float_data[i * dim + d];
                }
                neighbors[i] = {i, dot_product};
            }
        }

        partial_sort(neighbors.begin(), neighbors.begin() + K, neighbors.end());

        int hits = 0;
        for (int k = 0; k < K; ++k) {
            int my_prediction = neighbors[k].id;
            for (int gt = 0; gt < K + 1; ++gt) { 
                int official_id = gt_data[q * GT_K + gt] - 1; 
                if (official_id == q) continue;
                if (my_prediction == official_id) {
                    hits++; break;
                }
            }
        }
        total_recall += (float)hits / K;
    }
    auto algo_end = steady_clock::now();
    duration<double> algo_time = algo_end - algo_start;

    // 4. STOP GLOBAL TIMING
    duration<double> global_time = steady_clock::now() - global_start;
    float average_recall = total_recall / TEST_QUERIES;
    double peak_mb = get_peak_memory_mb();
    
    cout << "\n--- FINAL RESULTS ---" << endl;
    cout << "1. I/O Time:           " << io_time.count() << " seconds" << endl;
    if (mode != "fp32") cout << "2. Quantization Time:  " << quant_time.count() << " seconds" << endl;
    cout << "3. Search Time:        " << algo_time.count() << " seconds" << endl;
    cout << "--------------------------------------------------" << endl;
    cout << "TOTAL WALL-CLOCK:      " << global_time.count() << " seconds" << endl;
    cout << "--------------------------------------------------" << endl;
    cout << "Average Recall:        " << average_recall * 100.0f << "%" << endl;
    cout << "Peak RAM (VmPeak):     " << (peak_mb >= 1024.0 ? peak_mb / 1024.0 : peak_mb) 
         << (peak_mb >= 1024.0 ? " GB" : " MB") << endl;
    cout << "==================================================" << endl;

    return 0;
}