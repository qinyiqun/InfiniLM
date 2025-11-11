#include "bge.hpp"

#include "../../tensor.hpp"
#include "../../utils.hpp"
#include "../inference_context.hpp"

#include <random>
#include <thread>
#include <vector>

inline std::vector<int32_t>
position_ids_from_input_ids(const uint32_t *input_ids,
                            std::size_t seq_len,
                            std::size_t bsz,
                            uint32_t padding_idx) {
    std::vector<int32_t> pos;
    pos.reserve(bsz * seq_len); // 注意：总长度是 bsz * seq_len

    for (std::size_t b = 0; b < bsz; ++b) {
        int32_t cum = 0; // 每个 batch 独立累计
        for (std::size_t t = 0; t < seq_len; ++t) {
            std::size_t flat_idx = b * seq_len + t;
            int32_t m = (input_ids[flat_idx] != padding_idx);
            cum += m;
            // 保持原始逻辑：非 padding 位置为 (cum + past_kv_len)，padding 位置为 padding_idx
            // pos.push_back(cum * m + static_cast<int32_t>(padding_idx) * (1 - m));
            pos.push_back(cum * m + static_cast<int32_t>(padding_idx));
        }
    }

    return pos; // NRVO, no copy
}
// inline std::vector<int32_t>
// position_ids_from_input_ids(const uint32_t *input_ids,
//                             std::size_t seq_len,
//                             std::size_t bsz,
//                             uint32_t padding_idx) {
//     std::vector<int32_t> pos;
//     pos.reserve(seq_len);

//     int32_t cum = 0;
//     for (std::size_t i = 0; i < seq_len; ++i) {
//         int32_t m = (input_ids[i] != padding_idx);
//         cum += m;
//         pos.push_back((cum + past_kv_len) * m + static_cast<int32_t>(padding_idx));
//     }

//     return pos; // NRVO / move，无拷贝
// }

void inferDeviceBatch(const BGEM3Meta *meta, BGEM3DeviceResource &rsrc,
                      uint32_t idev, uint32_t ndev, uint32_t bsz,
                      const uint32_t *tokens, const float *masks, uint32_t ntok,
                      const uint32_t *req_lens, uint32_t nreq, const uint32_t *req_pos,
                      uint32_t *output, void *last_logits) {
    auto nlayer = meta->nlayer;
    auto nkvh = meta->nkvh / ndev;
    auto nh = meta->nh / ndev;
    auto ngroup = nh / nkvh;
    // auto dctx = meta.dctx;
    auto dh = meta->dh;
    auto d = meta->d;
    auto dt_logits = meta->dt_logits;
    auto di = meta->di / ndev;
    auto dvoc = meta->dvoc;
    auto stream = rsrc.stream;
    auto weight = rsrc.weights;

    // auto attn_masks = Tensor::buffer(dt_logits, {bsz, 1, ntok, ntok}, rsrc.memory_pool);
    std::shared_ptr<Tensor> attn_masks = Tensor::weight((void *)masks, INFINI_DTYPE_F32, {bsz, 1, ntok, ntok});
    auto logits_in = Tensor::buffer(dt_logits, {bsz, ntok, d}, rsrc.memory_pool);
    auto logits_in_copy = Tensor::buffer(dt_logits, {bsz, ntok, d}, rsrc.memory_pool);

    auto attention_mask = Tensor::buffer(dt_logits, {bsz, ntok, ntok}, rsrc.memory_pool);

    auto pos = position_ids_from_input_ids(tokens, ntok, bsz, 1);
    auto pos_buf = Tensor::buffer(dt_logits, {bsz, ntok, d}, rsrc.memory_pool);
    uint32_t token_type_ids[bsz * ntok] = {};
    auto token_type_buf = Tensor::buffer(dt_logits, {bsz, ntok, d}, rsrc.memory_pool);

    auto q_buf = Tensor::buffer(dt_logits, {bsz, ntok, nh * dh}, rsrc.memory_pool);
    auto k_buf = Tensor::buffer(dt_logits, {bsz, ntok, nkvh * dh}, rsrc.memory_pool);
    auto v_buf = Tensor::buffer(dt_logits, {bsz, ntok, nkvh * dh}, rsrc.memory_pool);

    auto qk_buf = Tensor::buffer(dt_logits, {bsz, nh, ntok, ntok}, rsrc.memory_pool);
    auto inter_buf = Tensor::buffer(dt_logits, {bsz, ntok, di}, rsrc.memory_pool);

    auto pooler = Tensor::buffer(dt_logits, {bsz, 1, d}, rsrc.memory_pool);
    auto sparse_out = Tensor::buffer(dt_logits, {bsz, ntok, 1}, rsrc.memory_pool);

    std::cout
        << "tnbl" << "dh: " << dh << "nlayer: " << nlayer << ", nh: " << nh << ", nkvh: " << nkvh << ", ngroup: " << ngroup << ", d: " << d << ", di: " << di << ", dvoc: " << dvoc << std::endl;

    // 计算XLM-Roberta的 embedding 层
    for (uint32_t b = 0; b < bsz; ++b) {
        for (uint32_t t = 0; t < ntok; ++t) {
            uint32_t flat_idx = b * ntok + t;           // flatten 后的 tokens 索引
            uint32_t output_offset = flat_idx * d;      // logits_in 中的偏移
            uint32_t emb_offset = tokens[flat_idx] * d; // embedding 表中的偏移

            RUN_INFINI(infinirtMemcpyAsync(
                logits_in->data(output_offset),
                weight->w_word_embd->data(emb_offset),
                dsize(dt_logits) * d,
                INFINIRT_MEMCPY_D2D,
                stream));
        }
    }

    // Position Embedding: pos is [bsz * ntok]
    for (uint32_t b = 0; b < bsz; ++b) {
        for (uint32_t t = 0; t < ntok; ++t) {
            uint32_t flat_idx = b * ntok + t;
            uint32_t output_offset = flat_idx * d;
            uint32_t pos_emb_offset = pos[flat_idx] * d;

            RUN_INFINI(infinirtMemcpyAsync(
                pos_buf->data(output_offset),
                weight->w_pos_embd->data(pos_emb_offset),
                dsize(dt_logits) * d,
                INFINIRT_MEMCPY_D2D,
                stream));
        }
    }

    // Token Type Embedding: token_type_ids is [bsz * ntok]
    for (uint32_t b = 0; b < bsz; ++b) {
        for (uint32_t t = 0; t < ntok; ++t) {
            uint32_t flat_idx = b * ntok + t;
            uint32_t output_offset = flat_idx * d;
            uint32_t tok_type_emb_offset = token_type_ids[flat_idx] * d;

            RUN_INFINI(infinirtMemcpyAsync(
                token_type_buf->data(output_offset),
                weight->w_tok_embd->data(tok_type_emb_offset),
                dsize(dt_logits) * d,
                INFINIRT_MEMCPY_D2D,
                stream));
        }
    }
    add(logits_in, logits_in, pos_buf);
    add(logits_in, logits_in, token_type_buf);
    layerNorm(logits_in, nullptr, nullptr, logits_in, weight->w_layer_norm, weight->b_layer_norm, 1e-5);

    // 接下来就是Attention层的计算
    for (uint32_t layer = 0; layer < nlayer; layer++) {
        rearrange(logits_in_copy, logits_in); // 保存一份输入用于残差连接

        linear(q_buf, logits_in, weight->w_attn_q[layer]->view_as({bsz, d, d}, {0, static_cast<ptrdiff_t>(d), 1})->permute({0, 2, 1}), 1.0, 0.0, nullptr, weight->b_attn_q[layer]);
        linear(k_buf, logits_in, weight->w_attn_k[layer]->view_as({bsz, d, d}, {0, static_cast<ptrdiff_t>(d), 1})->permute({0, 2, 1}), 1.0, 0.0, nullptr, weight->b_attn_k[layer]);
        linear(v_buf, logits_in, weight->w_attn_v[layer]->view_as({bsz, d, d}, {0, static_cast<ptrdiff_t>(d), 1})->permute({0, 2, 1}), 1.0, 0.0, nullptr, weight->b_attn_v[layer]);

        // for (size_t i = 0; i < (size_t)bsz; i++) {
        //     linear(qk_buf->slice({{0, i, 1}})->view_as({nh, ntok, ntok}),
        //            q_buf->view_as({bsz, q_buf->numel() / d / bsz, nh, d / nh})->slice({{0, i, 1}})->view_as({q_buf->numel() / d / bsz, nh, d / nh})->permute({1, 0, 2}),
        //            k_buf->view_as({bsz, k_buf->numel() / d / bsz, nh, d / nh})->slice({{0, i, 1}})->view_as({k_buf->numel() / d / bsz, nh, d / nh})->permute({1, 2, 0}),
        //            1.0f / std::sqrt(static_cast<float>(dh)), 0.0f, nullptr, nullptr);
        // }

        auto q_buf_copy = Tensor::buffer(dt_logits, {bsz, nh, ntok, d / nh}, rsrc.memory_pool);
        auto k_buf_copy = Tensor::buffer(dt_logits, {bsz, nh, ntok, d / nh}, rsrc.memory_pool);
        auto v_buf_copy = Tensor::buffer(dt_logits, {bsz, nh, ntok, d / nh}, rsrc.memory_pool);

        rearrange(q_buf_copy, q_buf->view_as({bsz, ntok, nh, d / nh})->permute({0, 2, 1, 3}));
        rearrange(k_buf_copy, k_buf->view_as({bsz, ntok, nh, d / nh})->permute({0, 2, 1, 3}));
        rearrange(v_buf_copy, v_buf->view_as({bsz, ntok, nh, d / nh})->permute({0, 2, 1, 3}));

        linear(qk_buf->view_as({bsz * nh, ntok, ntok}),
               q_buf_copy->view_as({nh * bsz, ntok, d / nh}),
               k_buf_copy->view_as({nh * bsz, ntok, d / nh})->permute({0, 2, 1}),
               1.0f / std::sqrt(static_cast<float>(dh)), 0.0f, nullptr, nullptr);

        add(qk_buf, qk_buf, attn_masks->view_as({bsz, nh, ntok, ntok}, {ntok * ntok, 0, ntok, 1}));
        softmax(qk_buf, qk_buf, -1);
        linear(v_buf->view_as({nh * bsz, ntok, d / nh}),
               qk_buf->view_as({bsz * nh, ntok, ntok}), v_buf_copy->view_as({nh * bsz, ntok, d / nh}),
               1.0, 0.0, nullptr, nullptr);

        rearrange(v_buf_copy->view_as({bsz, ntok, nh, d / nh}), v_buf->view_as({bsz, nh, ntok, d / nh})->permute({0, 2, 1, 3}));
        linear(logits_in, v_buf_copy->view_as({bsz, ntok, d}),
               weight->w_attn_out[layer]->view_as({bsz, d, d}, {0, static_cast<ptrdiff_t>(d), 1})->permute({0, 2, 1}),
               1.0, 0.0, nullptr, weight->b_attn_out[layer]);

        add(logits_in, logits_in, logits_in_copy); // residual connection
        layerNorm(logits_in, nullptr, nullptr, logits_in, weight->w_attn_layer_norm[layer], weight->b_attn_layer_norm[layer], 1e-5);
        // InterMediate Layer
        linear(inter_buf, logits_in,
               weight->w_intermediate[layer]->view_as({bsz, di, d}, {0, static_cast<ptrdiff_t>(d), 1})->permute({0, 2, 1}),
               1.0, 0.0, nullptr, weight->b_intermediate[layer]);
        gelu(inter_buf, inter_buf);

        linear(logits_in_copy, inter_buf,
               weight->w_out[layer]->view_as({bsz, d, di}, {0, static_cast<ptrdiff_t>(di), 1})->permute({0, 2, 1}),
               1.0, 0.0, nullptr, weight->b_out[layer]);

        add(logits_in, logits_in, logits_in_copy); // residual connection
        layerNorm(logits_in, nullptr, nullptr, logits_in, weight->w_out_layer_norm[layer], weight->b_out_layer_norm[layer], 1e-5);
    }

    linear(pooler, logits_in->slice({{1, 0, 1}}), weight->w_pooler->view_as({bsz, d, d}, {0, static_cast<ptrdiff_t>(d), 1})->permute({0, 2, 1}), 1.0, 0.0, nullptr, weight->b_pooler);
    tanh(pooler, pooler);

    linear(sparse_out, logits_in, weight->w_sparse->view_as({bsz, 1, d}, {0, static_cast<ptrdiff_t>(d), 1})->permute({0, 2, 1}), 1.0, 0.0, nullptr, weight->b_sparse);
    // sparse output
    relu(sparse_out, sparse_out);
    lpNorm(logits_in->slice({{1, 0, 1}}), logits_in->slice({{1, 0, 1}}), -1, 2, 1e-12);

    logits_in->slice({{1, 0, 1}})->debug();
    sparse_out->debug();

    // dense output
    // logits_in->slice({{1, 0, 1}})->debug();
    exit(0);
}

void createDeviceResource(BGEM3DeviceResource *rsrc, const BGEM3Meta *meta,
                          std::shared_ptr<BGEM3DeviceWeight> weights,
                          infiniDevice_t device, int idev,
                          int ndev, int dev_id,
                          infinicclComm_t comm) {
    RUN_INFINI(infinirtSetDevice(device, dev_id));
    infiniopHandle_t handle;
    infiniopCreateHandle(&handle);
    infinirtStream_t stream;
    infinirtStreamCreate(&stream);

    auto memory_pool = std::make_shared<MemoryPool>(128 * 1024 * 1024);

    *rsrc = BGEM3DeviceResource{
        device,
        dev_id,
        handle,
        weights,
        stream,
        comm,
        memory_pool,
    };
    RUN_INFINI(infinirtDeviceSynchronize());
}

void releaseDeviceResource(BGEM3DeviceResource &res) {
    infinirtDeviceSynchronize();
    // Release individual Tensors

    infiniopDestroyHandle(res.handle);
    res.handle = nullptr;
    infinirtStreamDestroy(res.stream);
    res.stream = nullptr;
    infinicclCommDestroy(res.comm);
    res.comm = nullptr;
}

__C void
inferBatchBGEM3(struct BGEM3Model *model, uint32_t bsz, const uint32_t *tokens, const float *masks, uint32_t ntok,
                const uint32_t *req_lens, uint32_t nreq, const uint32_t *req_pos,
                struct KVCache **kv_caches,
                const float *temperature, const uint32_t *topk, const float *topp,
                uint32_t *output) {
    model->req.tokens = tokens;
    model->req.bsz = bsz;
    model->req.masks = masks;
    model->req.ntok = ntok;
    model->req.req_lens = req_lens;
    model->req.nreq = nreq;
    model->req.req_pos = req_pos;
    model->req.kv_caches = kv_caches;
    model->req.output = output;
    model->req.logits = nullptr;
    model->req.temperature = temperature;
    model->req.topk = topk;
    model->req.topp = topp;

    for (size_t idev = 0; idev < model->dev_ids.size(); idev++) {
        std::unique_lock<std::mutex> lock(model->states[idev].mtx);
        model->states[idev].proceed = true;
        lock.unlock();
        model->states[idev].cv_start.notify_one();
    }
    for (size_t i = model->dev_ids.size(); i > 0; i--) {
        auto idev = i - 1;
        std::unique_lock<std::mutex> lock(model->states[idev].mtx);
        model->states[idev].cv_done.wait(lock, [&] { return !(model->states[idev].proceed); });
        lock.unlock();
    }
}

__C void
forwardBatchBGEM3(struct BGEM3Model *model, uint32_t bsz,
                  const uint32_t *tokens, const float *masks, uint32_t ntok,
                  const uint32_t *req_lens, uint32_t nreq, const uint32_t *req_pos,
                  struct KVCache **kv_caches,
                  void *logits) {
    model->req.tokens = tokens;
    model->req.bsz = bsz;
    model->req.masks = masks;
    model->req.ntok = ntok;
    model->req.req_lens = req_lens;
    model->req.nreq = nreq;
    model->req.req_pos = req_pos;
    model->req.kv_caches = kv_caches;
    model->req.output = nullptr;
    model->req.logits = logits;
    model->req.temperature = nullptr;
    model->req.topk = nullptr;
    model->req.topp = nullptr;

    for (size_t idev = 0; idev < model->dev_ids.size(); idev++) {
        std::unique_lock<std::mutex> lock(model->states[idev].mtx);
        model->states[idev].proceed = true;
        lock.unlock();
        model->states[idev].cv_start.notify_one();
    }
    for (size_t i = model->dev_ids.size(); i > 0; i--) {
        auto idev = i - 1;
        std::unique_lock<std::mutex> lock(model->states[idev].mtx);
        model->states[idev].cv_done.wait(lock, [&] { return !(model->states[idev].proceed); });
        lock.unlock();
    }
}

void launchDevice(const BGEM3Meta *meta, std::shared_ptr<BGEM3DeviceWeight> weights, BGEM3DeviceResource *rsrc, InferState &state, InferRequest &req,
                  infiniDevice_t device, int idev, int ndev, int dev_id, infinicclComm_t comm) {
    // Create Device Resource
    createDeviceResource(rsrc, meta, weights, device, idev, ndev, dev_id, comm);

    CacheManager cache_manager(100);
    InferenceContext ctx(rsrc->handle, rsrc->memory_pool, &cache_manager, rsrc->stream);

    // Set the inference context for this thread
    setInferenceContext(&ctx);

    {
        std::unique_lock<std::mutex> lock(state.mtx);
        state.loaded = true;
        lock.unlock();
        state.cv_load.notify_one();
    }

    // Infer Loop
    while (true) {
        std::unique_lock<std::mutex> lock(state.mtx);
        state.cv_start.wait(lock, [&] { return state.proceed || state.exit_flag; });
        // quit if exit_flag is set
        if (state.exit_flag) {
            break;
        }
        inferDeviceBatch(meta, *rsrc, idev, ndev, req.bsz, req.tokens, req.masks, req.ntok,
                         req.req_lens, req.nreq, req.req_pos, req.output, req.logits);

        state.proceed = false;
        lock.unlock();
        state.cv_done.notify_one();
    }

    // Clean-Up
    releaseDeviceResource(*rsrc);
    setInferenceContext(nullptr); // Clear the context when done
}

BGEM3Model::BGEM3Model(const BGEM3Meta *meta, const ModelWeights *weights_) {
    auto weights = (BGEM3Weights *)(weights_);
    device = weights->device();
    dev_ids = weights->devIds();
    int ndev = int(dev_ids.size());
    dev_resources = std::vector<BGEM3DeviceResource>(ndev);
    states = std::vector<InferState>(ndev);
    threads.resize(ndev);

    auto comms = std::vector<infinicclComm_t>(ndev, nullptr);
    if (ndev > 1) {
        RUN_INFINI(infinicclCommInitAll(device, comms.data(), ndev, dev_ids.data()));
    }
    for (int i = 0; i < ndev; i++) {
        threads[i] = std::thread(launchDevice, meta, weights->device_weights()[i], &dev_resources[i], std::ref(states[i]), std::ref(req), device, i, ndev, dev_ids[i], comms[i]);
    }
    for (int i = 0; i < ndev; i++) {
        std::unique_lock<std::mutex> lock(states[i].mtx);
        states[i].cv_load.wait(lock, [&] { return states[i].loaded; });
        lock.unlock();
    }
}

__C struct BGEM3Model *
createBGEM3Model(const BGEM3Meta *meta,
                 const ModelWeights *weights) {
    BGEM3Model *model = new BGEM3Model(meta, weights);
    return model;
}

__C void destroyBGEM3Model(struct BGEM3Model *model) {
    auto ndev = model->dev_resources.size();

    for (size_t idev = 0; idev < ndev; idev++) {
        std::unique_lock<std::mutex> lock(model->states[idev].mtx);
        model->states[idev].exit_flag = true;
        lock.unlock();
        model->states[idev].cv_start.notify_one();
    }

    for (size_t idev = 0; idev < ndev; idev++) {
        model->threads[idev].join();
    }

    delete model;
}