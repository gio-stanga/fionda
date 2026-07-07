#pragma once

#include "llama.h"
#include "common.h"

struct common_speculative;
struct common_sampler;

// comma separated list of all types
std::string common_speculative_type_name_str();

// convert string to type
enum common_speculative_type common_speculative_type_from_name(const std::string & name);

// convert type to string
std::string common_speculative_type_to_str(enum common_speculative_type type);

// check if the llama_context is compatible for speculative decoding
// note: clears the memory of the context
bool common_speculative_is_compat(llama_context * ctx_tgt);

common_speculative * common_speculative_init(
        common_params_speculative & params,
        llama_context             * ctx_tgt);

void common_speculative_free(common_speculative * spec);

// optionally call once at the beginning of a new generation
void common_speculative_begin(common_speculative * spec, const llama_tokens & prompt);

// sample up to n_draft tokens and add them to the batch using the draft model
llama_tokens common_speculative_draft(
                     common_speculative * spec,
        const common_params_speculative & params,
                     const llama_tokens & prompt,
                            llama_token   id_last);

// informs the speculative decoder that n_accepted tokens were accepted by the target model
void common_speculative_accept(common_speculative * spec, uint16_t n_accepted);

//
// tree drafting (eagle1, greedy only)
//
// the draft is a token tree instead of a chain: sibling branches are verified in a single
// target pass using per-token multi-sequence KV membership (a token attends only to its
// ancestors). leaf sequences 1..n_leaf_seqs are used as scratch; the trunk stays in seq 0.

#define COMMON_SPECULATIVE_TREE_MAX_LEAVES 64

struct common_speculative_tree_node {
    llama_token  token;
    int32_t      parent;    // index into nodes; -1 = child of id_last
    int32_t      depth;     // 0-based; verify position = n_past + 1 + depth
    llama_seq_id seq_first; // first leaf seq of this node's subtree (>= 1)

    std::vector<llama_seq_id> seqs; // leaf seqs of the subtree = verify-batch seq set
};

struct common_speculative_tree {
    // level order (parents before children); verify-batch row of node i is i + 1
    std::vector<common_speculative_tree_node> nodes;

    int32_t n_leaf_seqs = 0; // leaf seqs 1..n_leaf_seqs are used

    bool empty() const { return nodes.empty(); }
};

// draft a token tree (params.tree_branch topology). empty tree = not supported by the
// active implementation - fall back to common_speculative_draft
common_speculative_tree common_speculative_draft_tree(
                     common_speculative * spec,
        const common_params_speculative & params,
                     const llama_tokens & prompt,
                            llama_token   id_last);

// build the target verify batch for a drafted tree (row 0 = id_last @ n_past, row i+1 =
// nodes[i] @ n_past+1+depth, all logits) and prepare the target KV (trunk copied to the
// leaf seqs). the caller decodes the batch and then calls common_speculative_accept_tree
void common_speculative_tree_prepare_batch(
                          llama_context * ctx_tgt,
        const common_speculative_tree   & tree,
                            llama_token   id_last,
                              llama_pos   n_past,
                            llama_batch & batch_tgt);

// greedy tree verification after the target decode of the prepared batch. returns the
// accepted tokens (size >= 1; the last one is the bonus token sampled from the target).
// cleans the target KV so positions [0, n_past + result.size()) remain, in seq 0 only
llama_tokens common_speculative_accept_tree(
                     common_speculative * spec,
                         common_sampler * smpl,
                          llama_context * ctx_tgt,
        const common_speculative_tree   & tree,
                              llama_pos   n_past);

// print statistics about the speculative decoding
void common_speculative_print_stats(const common_speculative * spec);
