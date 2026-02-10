#pragma once

#include "llama.h"
#include "llama-mmap.h"

#include <cstdint>
#include <vector>
#include <set>

struct ggml_tensor;

// Forward declarations
struct llama_model;
struct llama_context;

/**
 * MoE Expert Prefetcher
 *
 * Provides async prefetching of expert weights for MoE models.
 * Uses madvise(WILLNEED) to hint the kernel to prefetch expert tensors
 * before they're needed.
 *
 * Integration with chunked routing:
 * - When processing layer N, prefetch layer N+1's experts
 * - Uses the routing decision from layer N to predict N+1's needs
 *   (exploiting routing similarity between adjacent layers)
 */
struct llama_moe_prefetcher {
    llama_moe_prefetcher() = default;
    ~llama_moe_prefetcher() = default;

    // Initialize with model's mmaps and expert tensors
    void init(
        const llama_mmaps & mmaps,
        int32_t n_layer,
        int32_t n_expert
    );

    // Register expert tensor pointers for a layer
    // Called during model load to record where expert weights live
    void register_layer_experts(
        int32_t layer_idx,
        const ggml_tensor * up_exps,
        const ggml_tensor * gate_exps,
        const ggml_tensor * down_exps
    );

    // Prefetch specific experts for the next layer
    // Called from eval callback when we see ffn_moe_topk computed
    void prefetch_experts_for_layer(
        int32_t layer_idx,
        const int32_t * expert_ids,
        int32_t n_experts_to_fetch
    );

    // Prefetch all experts for a layer (useful for pinned layers)
    void prefetch_all_experts_for_layer(int32_t layer_idx);

    // Set which layers are fungible (where prefetch lookahead is useful)
    void set_fungible_layers(const std::set<int> & layers);

    // Stats
    struct stats {
        uint64_t prefetch_calls = 0;
        uint64_t prefetch_bytes = 0;
        uint64_t layers_prefetched = 0;
    };

    stats get_stats() const { return m_stats; }
    void reset_stats() { m_stats = {}; }

private:
    struct layer_experts {
        const ggml_tensor * up_exps = nullptr;
        const ggml_tensor * gate_exps = nullptr;
        const ggml_tensor * down_exps = nullptr;
    };

    const llama_mmaps * m_mmaps = nullptr;
    std::vector<layer_experts> m_layer_experts;
    std::set<int> m_fungible_layers;
    int32_t m_n_expert = 0;
    stats m_stats;

    // Prefetch a single expert tensor slice
    void prefetch_expert_slice(const ggml_tensor * exps, int32_t expert_idx);
};

// Eval callback for prefetching
// user_data should be a llama_moe_prefetcher*
bool llama_moe_prefetch_eval_callback(struct ggml_tensor * t, bool ask, void * user_data);
