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
#include <sys/stat.h>

using namespace std;
using namespace std::chrono;
using namespace H5;

const int NUM_VECTORS = 200000;
const int DIM = 1024;
const int K_EVAL = 15;
const int K_BUILD = 48;
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

struct alignas(64) AlignedInt8 {
    int8_t data[DIM];
};

struct alignas(64) AlignedSketch {
    uint64_t data[16];
};

void build_tree(vector<int>& indices, vector<vector<int>>& leaves, mt19937& rng, const vector<AlignedInt8>& data) {
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
        
        const int8_t* v_idx = data[idx].data;
        const int8_t* v_p1 = data[p1].data;
        const int8_t* v_p2 = data[p2].data;
        
        #pragma omp simd reduction(+:d1, d2) aligned(v_idx, v_p1, v_p2: 64)
        for (int d = 0; d < DIM; ++d) {
            int8_t v = v_idx[d];
            d1 += v * v_p1[d];
            d2 += v * v_p2[d];
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

void write_string_attr(H5File& file, const string& name, const string& value) {
    StrType str_type(0, H5T_VARIABLE);
    DataSpace att_space(H5S_SCALAR);
    Attribute att = file.createAttribute(name, str_type, att_space);
    att.write(str_type, value);
}

void write_float_attr(H5File& file, const string& name, float value) {
    DataSpace att_space(H5S_SCALAR);
    Attribute att = file.createAttribute(name, PredType::NATIVE_FLOAT, att_space);
    att.write(PredType::NATIVE_FLOAT, &value);
}

int main(int argc, char** argv) {
    auto global_start = steady_clock::now();

    string filename = "";
    string task_name = "task1";

    for (int i = 1; i < argc; ++i) {
        string arg = argv[i];
        if (arg == "--dataset" && i + 1 < argc) {
            filename = argv[++i];
        } else if (arg == "--task" && i + 1 < argc) {
            task_name = argv[++i];
        } else if (filename.empty() && arg.find("--") != 0) {
            filename = arg;
        }
    }

    if (filename.empty()) {
        return -1;
    }

    vector<float> float_data;
    vector<AlignedInt8> int8_data;
    vector<AlignedSketch> bin_data;
    vector<Neighbor> knn_graph(NUM_VECTORS * K_BUILD);

    try {
        H5File file(filename, H5F_ACC_RDONLY);
        float_data.resize(NUM_VECTORS * DIM);
        file.openDataSet("/train").read(float_data.data(), PredType::NATIVE_FLOAT);
    } catch (const Exception &error) {
        return -1;
    }

    int8_data.resize(NUM_VECTORS);
    bin_data.resize(NUM_VECTORS);
    
    #pragma omp parallel for
    for (int i = 0; i < NUM_VECTORS; ++i) {
        for (int w = 0; w < 16; ++w) bin_data[i].data[w] = 0;
        for (int d = 0; d < DIM; ++d) {
            float scaled = float_data[i * DIM + d] * 127.0f;
            float clamped = max(-128.0f, min(127.0f, scaled));
            int8_data[i].data[d] = static_cast<int8_t>(clamped);
            
            if (clamped > 0) {
                bin_data[i].data[d / 64] |= (1ULL << (d % 64));
            }
        }
    }
    float_data.clear();
    float_data.shrink_to_fit();

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
        
        const int8_t* v1 = int8_data[i].data;
        for (int n2 : cands) {
            int32_t dot = 0;
            const int8_t* v2 = int8_data[n2].data;
            #pragma omp simd reduction(+:dot) aligned(v1, v2: 64)
            for (int d = 0; d < DIM; ++d) {
                dot += v1[d] * v2[d];
            }
            pool.push_back({n2, dot});
        }

        uint32_t rng_state = 1337 + i;
        while (pool.size() < K_BUILD) {
            rng_state ^= rng_state << 13;
            rng_state ^= rng_state >> 17;
            rng_state ^= rng_state << 5;
            int rand_id = rng_state % NUM_VECTORS;
            
            if (rand_id == i) continue;
            int32_t dot = 0;
            const int8_t* v2 = int8_data[rand_id].data;
            #pragma omp simd reduction(+:dot) aligned(v1, v2: 64)
            for (int d = 0; d < DIM; ++d) {
                dot += v1[d] * v2[d];
            }
            pool.push_back({rand_id, dot});
        }

        if (K_BUILD < pool.size()) {
            partial_sort(pool.begin(), pool.begin() + K_BUILD, pool.end());
        } else {
            sort(pool.begin(), pool.end());
        }

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
            rng_state ^= rng_state << 13;
            rng_state ^= rng_state >> 17;
            rng_state ^= rng_state << 5;
            int rand_id = rng_state % NUM_VECTORS;
            if (rand_id != i) {
                knn_graph[i * K_BUILD + added] = {rand_id, -999999};
                added++;
            }
        }
    }

    int max_iterations = 30;
    for (int iter = 0; iter < max_iterations; ++iter) {
        int total_updates = 0;

        #pragma omp parallel 
        {
            vector<Neighbor> local_pool;
            local_pool.reserve(K_BUILD);
            pair<int, int32_t> sketch_candidates[K_BUILD * K_BUILD];

            #pragma omp for reduction(+:total_updates)
            for (int i = 0; i < NUM_VECTORS; ++i) {
                local_pool.clear();
                int sketch_count = 0;
                
                for (int k = 0; k < K_BUILD; ++k) {
                    local_pool.push_back(knn_graph[i * K_BUILD + k]);
                }
                
                make_heap(local_pool.begin(), local_pool.end());

                const uint64_t* b1 = bin_data[i].data;

                for (int k = 0; k < K_BUILD; ++k) {
                    int n1 = knn_graph[i * K_BUILD + k].id;
                    for (int k2 = 0; k2 < K_BUILD; ++k2) {
                        if (k2 + 1 < K_BUILD) {
                            int n_next = knn_graph[n1 * K_BUILD + k2 + 1].id;
                            __builtin_prefetch(&bin_data[n_next].data[0], 0, 1);
                        }
                        
                        int n2 = knn_graph[n1 * K_BUILD + k2].id;
                        if (n2 == i) continue;

                        const uint64_t* b2 = bin_data[n2].data;

                        int32_t h_score = 1024;
                        h_score -= __builtin_popcountll(b1[0] ^ b2[0]);
                        h_score -= __builtin_popcountll(b1[1] ^ b2[1]);
                        h_score -= __builtin_popcountll(b1[2] ^ b2[2]);
                        h_score -= __builtin_popcountll(b1[3] ^ b2[3]);
                        h_score -= __builtin_popcountll(b1[4] ^ b2[4]);
                        h_score -= __builtin_popcountll(b1[5] ^ b2[5]);
                        h_score -= __builtin_popcountll(b1[6] ^ b2[6]);
                        h_score -= __builtin_popcountll(b1[7] ^ b2[7]);
                        h_score -= __builtin_popcountll(b1[8] ^ b2[8]);
                        h_score -= __builtin_popcountll(b1[9] ^ b2[9]);
                        h_score -= __builtin_popcountll(b1[10] ^ b2[10]);
                        h_score -= __builtin_popcountll(b1[11] ^ b2[11]);
                        h_score -= __builtin_popcountll(b1[12] ^ b2[12]);
                        h_score -= __builtin_popcountll(b1[13] ^ b2[13]);
                        h_score -= __builtin_popcountll(b1[14] ^ b2[14]);
                        h_score -= __builtin_popcountll(b1[15] ^ b2[15]);

                        sketch_candidates[sketch_count++] = {n2, h_score};
                    }
                }

                int top_m = sketch_count < K_BUILD * RERANK_MULTIPLIER ? sketch_count : K_BUILD * RERANK_MULTIPLIER;
                
                if (top_m > 0 && top_m < sketch_count) {
                    nth_element(sketch_candidates, sketch_candidates + top_m, sketch_candidates + sketch_count, [](const pair<int, int32_t>& a, const pair<int, int32_t>& b) {
                        return a.second > b.second;
                    });
                }

                const int8_t* v1 = int8_data[i].data;
                
                for (int m = 0; m < top_m; ++m) {
                    int n2 = sketch_candidates[m].first;
                    
                    bool exists = false;
                    for (size_t p = 0; p < K_BUILD; ++p) {
                        if (local_pool[p].id == n2) {
                            exists = true;
                            break;
                        }
                    }
                    if (exists) continue;

                    const int8_t* v2 = int8_data[n2].data;
                    int32_t dot = 0;
                    
                    #pragma omp simd reduction(+:dot) aligned(v1, v2: 64)
                    for (int d = 0; d < DIM; ++d) {
                        dot += v1[d] * v2[d];
                    }
                    
                    if (dot > local_pool.front().score) {
                        pop_heap(local_pool.begin(), local_pool.end());
                        local_pool.back() = {n2, dot};
                        push_heap(local_pool.begin(), local_pool.end());
                    }
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
        }
        
        float update_rate = (float)total_updates / (NUM_VECTORS * K_BUILD);
        if (update_rate < 0.05f) {
            break;
        }
    }
    auto algo_end = steady_clock::now();
    duration<double> build_time = algo_end - global_start;

    vector<int> final_knns(NUM_VECTORS * K_EVAL);
    vector<float> final_dists(NUM_VECTORS * K_EVAL);

    #pragma omp parallel for
    for (int i = 0; i < NUM_VECTORS; ++i) {
        for (int k = 0; k < K_EVAL; ++k) {
            final_knns[i * K_EVAL + k] = knn_graph[i * K_BUILD + k].id + 1;
            final_dists[i * K_EVAL + k] = static_cast<float>(knn_graph[i * K_BUILD + k].score);
        }
    }

    mkdir("results", 0777);
    string res_dir = "results/" + task_name;
    mkdir(res_dir.c_str(), 0777);
    string out_filename = res_dir + "/apex_simd.h5";

    try {
        H5File outfile(out_filename, H5F_ACC_TRUNC);

        hsize_t dims[2] = {static_cast<hsize_t>(NUM_VECTORS), static_cast<hsize_t>(K_EVAL)};
        DataSpace dataspace(2, dims);

        DataSet dataset_knns = outfile.createDataSet("knns", PredType::NATIVE_INT, dataspace);
        dataset_knns.write(final_knns.data(), PredType::NATIVE_INT);

        DataSet dataset_dists = outfile.createDataSet("dists", PredType::NATIVE_FLOAT, dataspace);
        dataset_dists.write(final_dists.data(), PredType::NATIVE_FLOAT);

        write_string_attr(outfile, "algo", "Apex Hardware SIMD");
        write_string_attr(outfile, "task", task_name);
        write_float_attr(outfile, "buildtime", static_cast<float>(build_time.count()));
        write_float_attr(outfile, "querytime", 0.0f);
        write_string_attr(outfile, "params", "K_BUILD=48, RERANK=8, TREES=8, SIMD=Aligned");

    } catch (const Exception &error) {
        return -1;
    }

    return 0;
}