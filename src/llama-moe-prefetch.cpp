#include "llama-moe-prefetch.h"

#include "ggml.h"

#include <cstring>
#include <string>

void llama_moe_prefetcher::init(
    const llama_mmaps & mmaps,
    int32_t n_layer,
    int32_t n_expert
) {
    m_mmaps = &mmaps;
    m_n_expert = n_expert;
    m_layer_experts.resize(n_layer);
}

void llama_moe_prefetcher::register_layer_experts(
    int32_t layer_idx,
    const ggml_tensor * up_exps,
    const ggml_tensor * gate_exps,
    const ggml_tensor * down_exps
) {
    if (layer_idx < 0 || layer_idx >= (int32_t)m_layer_experts.size()) {
        return;
    }
    m_layer_experts[layer_idx] = { up_exps, gate_exps, down_exps };
}

void llama_moe_prefetcher::set_fungible_layers(const std::set<int> & layers) {
    m_fungible_layers = layers;
}

void llama_moe_prefetcher::prefetch_expert_slice(const ggml_tensor * exps, int32_t expert_idx) {
    if (!exps || !exps->data || !m_mmaps) {
        return;
    }

    // Expert tensors are 3D: [dim0, dim1, n_expert]
    // Stride to next expert is nb[2]
    if (expert_idx < 0 || expert_idx >= m_n_expert) {
        return;
    }

    // Calculate pointer to this expert's data
    const char * expert_data = static_cast<const char *>(exps->data) + expert_idx * exps->nb[2];
    size_t expert_size = exps->nb[2];  // Size of one expert slice

    // Try each mmap to find which one contains this pointer
    for (const auto & mmap : *m_mmaps) {
        if (mmap->prefetch_ptr(expert_data, expert_size)) {
            m_stats.prefetch_bytes += expert_size;
            break;
        }
    }
}

void llama_moe_prefetcher::prefetch_experts_for_layer(
    int32_t layer_idx,
    const int32_t * expert_ids,
    int32_t n_experts_to_fetch
) {
    if (layer_idx < 0 || layer_idx >= (int32_t)m_layer_experts.size()) {
        return;
    }

    const auto & le = m_layer_experts[layer_idx];

    for (int32_t i = 0; i < n_experts_to_fetch; i++) {
        int32_t eid = expert_ids[i];
        prefetch_expert_slice(le.up_exps, eid);
        prefetch_expert_slice(le.gate_exps, eid);
        prefetch_expert_slice(le.down_exps, eid);
    }

    m_stats.prefetch_calls++;
    m_stats.layers_prefetched++;
}

void llama_moe_prefetcher::prefetch_all_experts_for_layer(int32_t layer_idx) {
    if (layer_idx < 0 || layer_idx >= (int32_t)m_layer_experts.size()) {
        return;
    }

    const auto & le = m_layer_experts[layer_idx];

    for (int32_t eid = 0; eid < m_n_expert; eid++) {
        prefetch_expert_slice(le.up_exps, eid);
        prefetch_expert_slice(le.gate_exps, eid);
        prefetch_expert_slice(le.down_exps, eid);
    }

    m_stats.prefetch_calls++;
    m_stats.layers_prefetched++;
}

// Parse layer index from tensor name like "ffn_moe_topk-12"
static int parse_layer_from_name(const char * name) {
    if (!name) return -1;

    // Find the last dash
    const char * dash = strrchr(name, '-');
    if (!dash) return -1;

    // Parse the number after the dash
    char * end = nullptr;
    long layer = strtol(dash + 1, &end, 10);
    if (end == dash + 1 || *end != '\0') {
        return -1;
    }
    return (int)layer;
}

bool llama_moe_prefetch_eval_callback(struct ggml_tensor * t, bool ask, void * user_data) {
    // When ask=true, scheduler is asking if we want to observe this tensor
    // When ask=false, tensor computation is complete and we can read its data

    if (!user_data || !t) {
        return true;  // Continue execution
    }

    // We only care about ffn_moe_topk tensors (routing decisions)
    if (strncmp(t->name, "ffn_moe_topk", 12) != 0) {
        return true;
    }

    // Skip the chunked version - we want the original routing
    if (strstr(t->name, "chunked") != nullptr) {
        return true;
    }

    llama_moe_prefetcher * prefetcher = static_cast<llama_moe_prefetcher *>(user_data);

    if (ask) {
        // Yes, we want to observe this tensor
        return true;
    }

    // Tensor is computed - use its routing to prefetch NEXT layer
    int layer_idx = parse_layer_from_name(t->name);
    if (layer_idx < 0) {
        return true;
    }

    // Prefetch for layer_idx + 1 (lookahead)
    int next_layer = layer_idx + 1;

    // Only prefetch for fungible layers (where we benefit from lookahead)
    // For pinned layers, everything is already in VRAM

    // Read the expert indices from the tensor
    // Shape: [n_expert_used, n_tokens]
    // We just need the unique experts from the first token (or first chunk leader)
    if (!t->data) {
        return true;
    }

    int32_t n_expert_used = (int32_t)t->ne[0];
    // int32_t n_tokens = (int32_t)t->ne[1];

    // Get expert IDs from first token
    const int32_t * expert_ids = static_cast<const int32_t *>(t->data);

    prefetcher->prefetch_experts_for_layer(next_layer, expert_ids, n_expert_used);

    return true;  // Continue execution
}
