#pragma once
#include "infinicore_infer/models/bge_reranker.h"

#include "../../cache.hpp"
#include "../../dataloader/weights_loader.hpp"

#include <condition_variable>
#include <mutex>
#include <thread>

struct BGERerankerDeviceWeight {
    std::shared_ptr<Tensor> w_word_embd, w_pos_embd, w_tok_embd, w_layer_norm, b_layer_norm, w_dense_cls, b_dense_cls, w_out_cls, b_out_cls;
    std::vector<std::shared_ptr<Tensor>> w_attn_q, b_attn_q, w_attn_k, b_attn_k, w_attn_v, b_attn_v, w_attn_out, b_attn_out, w_attn_layer_norm, b_attn_layer_norm, w_intermediate, b_intermediate, w_out, b_out, w_out_layer_norm, b_out_layer_norm;
};

class BGERerankerWeights : public infinicore::weights::Loader {
private:
    std::vector<std::shared_ptr<BGERerankerDeviceWeight>> _device_weights;

public:
    BGERerankerWeights(const BGERerankerMeta *meta,
                       infiniDevice_t device,
                       const std::vector<int> &dev_ids);
    std::vector<std::shared_ptr<BGERerankerDeviceWeight>> &device_weights() {
        return _device_weights;
    }
};

struct BGERerankerDeviceResource {
    // Device
    infiniDevice_t device;
    int device_id;
    infiniopHandle_t handle;
    // Weights
    std::shared_ptr<BGERerankerDeviceWeight> weights;
    // Streams
    infinirtStream_t stream;
    // Communicator
    infinicclComm_t comm;

    std::shared_ptr<MemoryPool> memory_pool;
};

struct InferRequest {
    uint32_t bsz;
    const uint32_t *tokens;
    const void *masks;
    uint32_t ntok;
    void *dense_out;
    void *sparse_out;
};

struct InferState {
    std::mutex mtx;
    std::condition_variable cv_load, cv_start, cv_done;
    bool loaded = false;
    bool proceed = false;
    bool exit_flag = false;
};

struct BGERerankerModel {
    BGERerankerMeta meta;
    infiniDevice_t device;
    std::vector<int> dev_ids;
    std::vector<BGERerankerDeviceResource> dev_resources;
    std::vector<InferState> states;
    std::vector<std::thread> threads;
    InferRequest req;

    BGERerankerModel(const BGERerankerMeta *, const ModelWeights *);
};