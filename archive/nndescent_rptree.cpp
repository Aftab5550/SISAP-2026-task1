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
const int NUM_TREES = 8;
const int LEAF_SIZE = 64;

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

void build_tree(vector<int>& indices, vector<vector<int>>& leaves, mt19937& rng, const vector<int8_t>& data) {
    if (indices.size() <= LEAF_SIZE) {
        #pragma omp critical
        leaves.push_back(indices);
        return;
    }
    
    uniform_int_distribution<int> dist(0, indices.size() - 1);
    int p1 = indices[dist(rng)];
    int p2 = indices[dist(rng)];
    while (p1 == p2) {
        p2 = indices[dist(rng)];
    }

    vector<int> left;
    vector<int> right;
    left.reserve(indices.size());
    right.reserve(indices.size());

    for (int idx : indices) {
        if (idx == p1) {
            left.push_back(idx);
            continue;
        }
        if (idx == p2) {
            right.push_back(idx);
            continue;
        }
        
        int32_t d1 = 0;
        int32_t d2 = 0;
        for (int d = 0; d < DIM; ++d) {
            int8_t v = data[idx * DIM + d];
            d1 += v * data[p1 * DIM + d];
            d2 += v * data[p2 * DIM + d];
        }
        
        if (d1 > d2) {
            left.push_back(idx);
        } else {
            right.push_back(idx);
        }
    }

    if (left.empty()) {
        left.push_back(right.back());
        right.pop_back();
    }
    if (right.empty()) {
        right.push_back(left.back());
        left.pop_back();
    }

    build_tree(left, leaves, rng, data);
    build_tree(right, leaves, rng, data);
}

int main(int argc, char** argv) {
    auto global_start = steady_clock::now();

    if (argc != 2) {
        cerr << "Usage: ./nndescent_rptree <h5_file>\n";
        return -1;
    }

    string filename = argv[1];

    cout << "==================================================\n";
    cout << "   SISAP 2026: RP-TREE + BITWISE GRAPH ENGINE     \n";
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
    cout << "[3] Building RP-Tree Forest (Trees=" << NUM_TREES << ")...\n";

    vector<vector<int>> leaves;
    #pragma omp parallel for
    for (int t = 0; t < NUM_TREES; ++t) {
        mt19937 rng(1337 + t);
        vector<int> initial_indices(NUM_VECTORS);
        for (int i = 0; i < NUM_VECTORS; ++i) {
            initial_indices[i] = i;
        }
        build_tree(initial_indices, leaves, rng, int8_data);
    }

    vector<vector<int>> all_candidates(NUM_VECTORS);
    for (const auto& leaf : leaves) {
        for (int i : leaf) {
            for (int j : leaf) {
                if (i != j) {
                    all_candidates[i].push_back(j);
                }
            }
        }
    }

    #pragma omp parallel for
    for (int i = 0; i < NUM_VECTORS; ++i) {
        vector<int>& cands = all_candidates[i];
        sort(cands.begin(), cands.end());
        cands.erase(unique(cands.begin(), cands.end()), cands.end());

        vector<Neighbor> pool;
        pool.reserve(cands.size() + K_BUILD);
        for (int n2 : cands) {
            int32_t dot = 0;
            for (int d = 0; d < DIM; ++d) {
                dot += int8_data[i * DIM + d] * int8_data[n2 * DIM + d];
            }
            pool.push_back({n2, dot});
        }

        mt19937 rng(1337 + i);
        uniform_int_distribution<int> uni(0, NUM_VECTORS - 1);
        while (pool.size() < K_BUILD) {
            int rand_id = uni(rng);
            if (rand_id == i) continue;
            int32_t dot = 0;
            for (int d = 0; d < DIM; ++d) {
                dot += int8_data[i * DIM + d] * int8_data[rand_id * DIM + d];
            }
            pool.push_back({rand_id, dot});
        }

        sort(pool.begin(), pool.end());

        int added = 0;
        int last_id = -1;
        for (size_t p = 0; p < pool.size() && added < K_BUILD; ++p) {
            if (pool[p].id != last_id) {
                knn_graph[i * K_BUILD + added] = pool[p];
                last_id = pool[p].id;
                added++;
            }
        }
        while (added < K_BUILD) {
            int rand_id = uni(rng);
            if (rand_id != i) {
                knn_graph[i * K_BUILD + added] = {rand_id, -999999};
                added++;
            }
        }
    }
    auto init_end = steady_clock::now();

    float initial_recall = evaluate_recall(knn_graph, gt_data);
    cout << "    -> Forest Recall (Top 15): " << initial_recall * 100.0f << "%\n";

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

    cout << "\n--- FINAL RP-TREE + BITWISE RESULTS ---\n";
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