#pragma once
#include "infinicore_infer/models/bge.h"

#include "../../cache.hpp"
#include "../../dataloader/weights_loader.hpp"

#include <condition_variable>
#include <mutex>
#include <thread>

struct BGEM3DeviceWeight {
    std::shared_ptr<Tensor> w_word_embd, w_pos_embd, w_tok_embd, w_layer_norm, b_layer_norm, w_pooler, b_pooler, w_sparse, b_sparse, w_colbert, b_colbert;
    std::vector<std::shared_ptr<Tensor>> w_attn_q, b_attn_q, w_attn_k, b_attn_k, w_attn_v, b_attn_v, w_attn_out, b_attn_out, w_attn_layer_norm, b_attn_layer_norm, w_intermediate, b_intermediate, w_out, b_out, w_out_layer_norm, b_out_layer_norm;
};

class BGEM3Weights : public infinicore::weights::Loader {
private:
    std::vector<std::shared_ptr<BGEM3DeviceWeight>> _device_weights;

public:
    BGEM3Weights(const BGEM3Meta *meta,
                 infiniDevice_t device,
                 const std::vector<int> &dev_ids);
    std::vector<std::shared_ptr<BGEM3DeviceWeight>> &device_weights() {
        return _device_weights;
    }
};

struct BGEM3DeviceResource {
    // Device
    infiniDevice_t device;
    int device_id;
    infiniopHandle_t handle;
    // Weights
    std::shared_ptr<BGEM3DeviceWeight> weights;
    // Streams
    infinirtStream_t stream;
    // Communicator
    infinicclComm_t comm;

    std::shared_ptr<MemoryPool> memory_pool;
};

struct InferRequest {
    uint32_t bsz;
    const uint32_t *tokens;
    const float *masks;
    uint32_t ntok;
    const uint32_t *req_lens;
    uint32_t nreq;
    const uint32_t *req_pos;
    struct KVCache **kv_caches;
    const float *temperature;
    const uint32_t *topk;
    const float *topp;
    float *dense_out;
    float *sparse_out;
};

struct InferState {
    std::mutex mtx;
    std::condition_variable cv_load, cv_start, cv_done;
    bool loaded = false;
    bool proceed = false;
    bool exit_flag = false;
};

struct BGEM3Model {
    BGEM3Meta meta;
    infiniDevice_t device;
    std::vector<int> dev_ids;
    std::vector<BGEM3DeviceResource> dev_resources;
    std::vector<InferState> states;
    std::vector<std::thread> threads;
    InferRequest req;

    BGEM3Model(const BGEM3Meta *, const ModelWeights *);
};