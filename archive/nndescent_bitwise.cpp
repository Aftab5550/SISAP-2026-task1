#include <iostream>
#include <vector>
#include <chrono>
#include <string>
#include <cmath>
#include <cstdint>
#include <algorithm>
#include <random>
#include <omp.h>
#include <H5Cpp.h>
#include <utility>

using namespace std;
using namespace std::chrono;
using namespace H5;

const int NUM_VECTORS = 200000;
const int DIM = 1024;
const int K_EVAL = 15;
const int K_BUILD = 48;
const int GT_K = 32;
const int RERANK_MULTIPLIER = 8;

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
        for (int k = 0; k < K_EVAL; ++k) {
            int my_pred = graph[i * K_BUILD + k].id;
            for (int gt = 0; gt < K_EVAL + 1; ++gt) {
                int official_id = gt_data[i * GT_K + gt] - 1;
                if (official_id == i) continue;
                if (my_pred == official_id) {
                    hits++;
                    break;
                }
            }
        }
    }
    return (float)hits / (NUM_VECTORS * K_EVAL);
}

int main(int argc, char** argv) {
    auto global_start = steady_clock::now();

    if (argc != 2) {
        cerr << "Usage: ./nndescent_bitwise <h5_file>\n";
        return -1;
    }

    string filename = argv[1];

    cout << "==================================================\n";
    cout << "   SISAP 2026: BITWISE SKETCH NN-DESCENT ENGINE   \n";
    cout << "==================================================\n";

    vector<float> float_data;
    vector<int8_t> int8_data;
    vector<uint64_t> bin_data;
    vector<Neighbor> knn_graph(NUM_VECTORS * K_BUILD);
    vector<int> gt_data(NUM_VECTORS * GT_K);

    auto io_start = steady_clock::now();
    cout << "[1] Loading HDF5 data into RAM...\n";
    try {
        H5File file(filename, H5F_ACC_RDONLY);
        float_data.resize(NUM_VECTORS * DIM);
        file.openDataSet("/train").read(float_data.data(), PredType::NATIVE_FLOAT);
        file.openDataSet("/allknn/knns").read(gt_data.data(), PredType::NATIVE_INT);
    } catch (const Exception &error) {
        cerr << "Error reading HDF5 file!\n";
        return -1;
    }
    auto io_end = steady_clock::now();

    auto quant_start = steady_clock::now();
    cout << "[2] Applying Int8 and 1-Bit Quantization...\n";
    int8_data.resize(NUM_VECTORS * DIM);
    bin_data.resize(NUM_VECTORS * 16, 0);
    
    #pragma omp parallel for
    for (int i = 0; i < NUM_VECTORS; ++i) {
        for (int d = 0; d < DIM; ++d) {
            float scaled = float_data[i * DIM + d] * 127.0f;
            if (scaled > 127.0f) scaled = 127.0f;
            if (scaled < -128.0f) scaled = -128.0f;
            int8_data[i * DIM + d] = static_cast<int8_t>(round(scaled));
            
            if (scaled > 0) {
                bin_data[i * 16 + (d / 64)] |= (1ULL << (d % 64));
            }
        }
    }
    float_data.clear();
    float_data.shrink_to_fit();
    auto quant_end = steady_clock::now();

    auto init_start = steady_clock::now();
    cout << "[3] Initializing Oversampled Graph (K=" << K_BUILD << ")...\n";

    #pragma omp parallel
    {
        mt19937 rng(1337 + omp_get_thread_num());
        uniform_int_distribution<int> uni(0, NUM_VECTORS - 1);

        #pragma omp for
        for (int i = 0; i < NUM_VECTORS; ++i) {
            for (int k = 0; k < K_BUILD; ++k) {
                int rand_id;
                do {
                    rand_id = uni(rng);
                } while (rand_id == i);
                
                int32_t dot = 0;
                for (int d = 0; d < DIM; ++d) {
                    dot += int8_data[i * DIM + d] * int8_data[rand_id * DIM + d];
                }
                knn_graph[i * K_BUILD + k] = {rand_id, dot};
            }
            sort(knn_graph.begin() + i * K_BUILD, knn_graph.begin() + i * K_BUILD + K_BUILD);
        }
    }
    auto init_end = steady_clock::now();

    float initial_recall = evaluate_recall(knn_graph, gt_data);
    cout << "    -> Initial Recall (Top 15): " << initial_recall * 100.0f << "%\n";

    cout << "[4] Starting Bitwise Dynamic NN-Descent...\n";
    auto algo_start = steady_clock::now();

    int max_iterations = 30;
    for (int iter = 0; iter < max_iterations; ++iter) {
        auto iter_start = steady_clock::now();
        int total_updates = 0;

        #pragma omp parallel for reduction(+:total_updates)
        for (int i = 0; i < NUM_VECTORS; ++i) {
            vector<Neighbor> local_pool;
            local_pool.reserve(K_BUILD + K_BUILD * RERANK_MULTIPLIER);
            
            for (int k = 0; k < K_BUILD; ++k) {
                local_pool.push_back(knn_graph[i * K_BUILD + k]);
            }

            vector<pair<int, int32_t>> sketch_candidates;
            sketch_candidates.reserve(K_BUILD * K_BUILD);

            for (int k = 0; k < K_BUILD; ++k) {
                int n1 = knn_graph[i * K_BUILD + k].id;
                for (int k2 = 0; k2 < K_BUILD; ++k2) {
                    int n2 = knn_graph[n1 * K_BUILD + k2].id;
                    if (n2 == i) continue;

                    __builtin_prefetch(&bin_data[n2 * 16], 0, 1);

                    int32_t h_score = 1024;
                    for(int w = 0; w < 16; ++w) {
                        h_score -= __builtin_popcountll(bin_data[i * 16 + w] ^ bin_data[n2 * 16 + w]);
                    }
                    sketch_candidates.push_back({n2, h_score});
                }
            }

            sort(sketch_candidates.begin(), sketch_candidates.end(), [](const pair<int, int32_t>& a, const pair<int, int32_t>& b) {
                return a.second > b.second;
            });

            int top_m = sketch_candidates.size() < K_BUILD * RERANK_MULTIPLIER ? sketch_candidates.size() : K_BUILD * RERANK_MULTIPLIER;
            for (int m = 0; m < top_m; ++m) {
                int n2 = sketch_candidates[m].first;
                
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

            sort(local_pool.begin(), local_pool.end());
            
            int local_changes = 0;
            for (int k = 0; k < K_BUILD; ++k) {
                if (knn_graph[i * K_BUILD + k].id != local_pool[k].id) {
                    knn_graph[i * K_BUILD + k] = local_pool[k];
                    local_changes++;
                }
            }
            total_updates += local_changes;
        }
        
        auto iter_end = steady_clock::now();
        duration<double> iter_time = iter_end - iter_start;
        float current_recall = evaluate_recall(knn_graph, gt_data);
        
        cout << "    -> Iteration " << iter + 1 << " | Updates: " << total_updates 
             << " | Recall: " << current_recall * 100.0f << "% | Time: " << iter_time.count() << "s\n";
             
        float update_rate = (float)total_updates / (NUM_VECTORS * K_BUILD);
        if (update_rate < 0.02f) {
            cout << "    -> Early Stopping Triggered (Update rate < 2%)\n";
            break;
        }
    }
    auto algo_end = steady_clock::now();

    duration<double> global_time = steady_clock::now() - global_start;

    cout << "\n--- FINAL BITWISE GRAPH RESULTS ---\n";
    cout << "1. I/O Time:           " << duration<double>(io_end - io_start).count() << " seconds\n";
    cout << "2. Quantization Time:  " << duration<double>(quant_end - quant_start).count() << " seconds\n";
    cout << "3. Graph Init Time:    " << duration<double>(init_end - init_start).count() << " seconds\n";
    cout << "4. Optimization Time:  " << duration<double>(algo_end - algo_start).count() << " seconds\n";
    cout << "--------------------------------------------------\n";
    cout << "TOTAL WALL-CLOCK:      " << global_time.count() << " seconds\n";
    cout << "Final Graph Recall:    " << evaluate_recall(knn_graph, gt_data) * 100.0f << "%\n";
    cout << "==================================================\n";

    return 0;
}