#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <string>
#include <cmath>
#include <cstdint>
#include <algorithm>
#include <omp.h>
#include <H5Cpp.h>

using namespace std;
using namespace std::chrono;
using namespace H5;

const int NUM_VECTORS = 200000;
const int DIM = 1024;
const int K = 15;
const int GT_K = 32;

struct Neighbor {
    int id;
    int32_t score;
    bool operator<(const Neighbor& other) const {
        return score > other.score;
    }
};

float evaluate_recall(const vector<Neighbor>& graph, const vector<int>& gt_data) {
    int hits = 0;
    for (int i = 0; i < NUM_VECTORS; ++i) {
        for (int k = 0; k < K; ++k) {
            int my_pred = graph[i * K + k].id;
            for (int gt = 0; gt < K + 1; ++gt) {
                int official_id = gt_data[i * GT_K + gt] - 1;
                if (official_id == i) continue;
                if (my_pred == official_id) {
                    hits++;
                    break;
                }
            }
        }
    }
    return (float)hits / (NUM_VECTORS * K);
}

int main(int argc, char** argv) {
    auto global_start = steady_clock::now();

    if (argc != 2) {
        cerr << "Usage: ./nndescent <h5_file>" << endl;
        return -1;
    }

    string filename = argv[1];

    cout << "==================================================" << endl;
    cout << "   SISAP 2026: NN-DESCENT GRAPH ENGINE            " << endl;
    cout << "==================================================" << endl;

    vector<float> float_data;
    vector<int8_t> int8_data;
    vector<Neighbor> knn_graph(NUM_VECTORS * K);
    vector<int> gt_data(NUM_VECTORS * GT_K);

    auto io_start = steady_clock::now();
    cout << "[1] Loading HDF5 data into RAM..." << endl;
    try {
        H5File file(filename, H5F_ACC_RDONLY);
        float_data.resize(NUM_VECTORS * DIM);
        file.openDataSet("/train").read(float_data.data(), PredType::NATIVE_FLOAT);
        file.openDataSet("/allknn/knns").read(gt_data.data(), PredType::NATIVE_INT);
    } catch (const Exception &error) {
        cerr << "Error reading HDF5 file!" << endl;
        return -1;
    }
    auto io_end = steady_clock::now();

    auto quant_start = steady_clock::now();
    cout << "[2] Applying Int8 Quantization..." << endl;
    int8_data.resize(NUM_VECTORS * DIM);
    
    #pragma omp parallel for
    for (size_t i = 0; i < float_data.size(); ++i) {
        float scaled = float_data[i] * 127.0f;
        if (scaled > 127.0f) scaled = 127.0f;
        if (scaled < -128.0f) scaled = -128.0f;
        int8_data[i] = static_cast<int8_t>(round(scaled));
    }
    float_data.clear();
    float_data.shrink_to_fit();
    auto quant_end = steady_clock::now();

    auto init_start = steady_clock::now();
    cout << "[3] Initializing Graph & Calculating Initial Distances..." << endl;

    #pragma omp parallel
    {
        mt19937 rng(1337 + omp_get_thread_num());
        uniform_int_distribution<int> uni(0, NUM_VECTORS - 1);

        #pragma omp for
        for (int i = 0; i < NUM_VECTORS; ++i) {
            for (int k = 0; k < K; ++k) {
                int rand_id;
                do {
                    rand_id = uni(rng);
                } while (rand_id == i);
                
                int32_t dot = 0;
                for (int d = 0; d < DIM; ++d) {
                    dot += int8_data[i * DIM + d] * int8_data[rand_id * DIM + d];
                }
                knn_graph[i * K + k] = {rand_id, dot};
            }
            sort(knn_graph.begin() + i * K, knn_graph.begin() + i * K + K);
        }
    }
    auto init_end = steady_clock::now();

    float initial_recall = evaluate_recall(knn_graph, gt_data);
    cout << "    -> Initial Random Recall: " << initial_recall * 100.0f << "%" << endl;

    cout << "[4] Starting NN-Descent Optimization Loop..." << endl;
    auto algo_start = steady_clock::now();

    int max_iterations = 30;
    for (int iter = 0; iter < max_iterations; ++iter) {
        auto iter_start = steady_clock::now();
        vector<Neighbor> old_graph = knn_graph;
        int total_updates = 0;

        #pragma omp parallel for reduction(+:total_updates)
        for (int i = 0; i < NUM_VECTORS; ++i) {
            vector<Neighbor> local_pool;
            for (int k = 0; k < K; ++k) {
                local_pool.push_back(old_graph[i * K + k]);
            }

            for (int k = 0; k < K; ++k) {
                int n1 = old_graph[i * K + k].id;
                for (int k2 = 0; k2 < K; ++k2) {
                    int n2 = old_graph[n1 * K + k2].id;
                    if (n2 == i) continue;

                    bool exists = false;
                    for (size_t p = 0; p < local_pool.size(); ++p) {
                        if (local_pool[p].id == n2) {
                            exists = true;
                            break;
                        }
                    }
                    if (exists) continue;

                    int32_t dot = 0;
                    for (int d = 0; d < DIM; ++d) {
                        dot += int8_data[i * DIM + d] * int8_data[n2 * DIM + d];
                    }
                    
                    local_pool.push_back({n2, dot});
                }
            }

            sort(local_pool.begin(), local_pool.end());
            
            int local_changes = 0;
            for (int k = 0; k < K; ++k) {
                if (knn_graph[i * K + k].id != local_pool[k].id) {
                    knn_graph[i * K + k] = local_pool[k];
                    local_changes++;
                }
            }
            total_updates += local_changes;
        }
        
        auto iter_end = steady_clock::now();
        duration<double> iter_time = iter_end - iter_start;
        float current_recall = evaluate_recall(knn_graph, gt_data);
        
        cout << "    -> Iteration " << iter + 1 << " | Updates: " << total_updates 
             << " | Recall: " << current_recall * 100.0f << "% | Time: " << iter_time.count() << "s" << endl;
             
        if (total_updates == 0) break;
    }
    auto algo_end = steady_clock::now();

    duration<double> global_time = steady_clock::now() - global_start;

    cout << "\n--- FINAL NN-DESCENT RESULTS ---" << endl;
    cout << "1. I/O Time:           " << duration<double>(io_end - io_start).count() << " seconds" << endl;
    cout << "2. Quantization Time:  " << duration<double>(quant_end - quant_start).count() << " seconds" << endl;
    cout << "3. Graph Init Time:    " << duration<double>(init_end - init_start).count() << " seconds" << endl;
    cout << "4. Optimization Time:  " << duration<double>(algo_end - algo_start).count() << " seconds" << endl;
    cout << "--------------------------------------------------" << endl;
    cout << "TOTAL WALL-CLOCK:      " << global_time.count() << " seconds" << endl;
    cout << "Final Graph Recall:    " << evaluate_recall(knn_graph, gt_data) * 100.0f << "%" << endl;
    cout << "==================================================" << endl;

    return 0;
}