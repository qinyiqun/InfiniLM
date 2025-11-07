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
                            uint32_t padding_idx,
                            int32_t past_kv_len = 0) {
    std::vector<int32_t> pos;
    pos.reserve(seq_len);

    int32_t cum = 0;
    for (std::size_t i = 0; i < seq_len; ++i) {
        int32_t m = (input_ids[i] != padding_idx);
        cum += m;
        pos.push_back((cum + past_kv_len) * m + static_cast<int32_t>(padding_idx));
    }
    return pos; // NRVO / move，无拷贝
}

void inferDeviceBatch(const BGEMeta *meta, BGEDeviceResource &rsrc,
                      uint32_t idev, uint32_t ndev,
                      const uint32_t *tokens, uint32_t ntok,
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

    auto logits_in = Tensor::buffer(dt_logits, {ntok, d}, rsrc.memory_pool);
    auto logits_in_copy = Tensor::buffer(dt_logits, {ntok, d}, rsrc.memory_pool);

    auto pos = position_ids_from_input_ids(tokens, ntok, 1);

    auto pos_buf = Tensor::buffer(dt_logits, {ntok, d}, rsrc.memory_pool);
    uint32_t token_type_ids[ntok] = {};
    auto token_type_buf = Tensor::buffer(dt_logits, {ntok, d}, rsrc.memory_pool);

    auto q_buf = Tensor::buffer(dt_logits, {ntok, nh * dh}, rsrc.memory_pool);
    auto k_buf = Tensor::buffer(dt_logits, {ntok, nkvh * dh}, rsrc.memory_pool);
    auto v_buf = Tensor::buffer(dt_logits, {ntok, nkvh * dh}, rsrc.memory_pool);
    auto qk_buf = Tensor::buffer(dt_logits, {nh, v_buf->numel() / d, v_buf->numel() / d}, rsrc.memory_pool);

    auto inter_buf = Tensor::buffer(dt_logits, {ntok, di}, rsrc.memory_pool);
    auto pooler = Tensor::buffer(dt_logits, {1, d}, rsrc.memory_pool);

    auto sparse_out = Tensor::buffer(dt_logits, {ntok, 1}, rsrc.memory_pool);

    std::cout
        << "tnbl" << "dh: " << dh << "nlayer: " << nlayer << ", nh: " << nh << ", nkvh: " << nkvh << ", ngroup: " << ngroup << ", d: " << d << ", di: " << di << ", dvoc: " << dvoc << std::endl;
    for (uint32_t i = 0; i < ntok; i++) {
        std::cout << "token[" << i << "]: " << tokens[i] << std::endl;
    }

    // 计算XLM-Roberta的 embedding 层
    for (uint32_t i = 0; i < ntok; i++) {
        RUN_INFINI(infinirtMemcpyAsync(logits_in->data(i * d),
                                       weight->w_word_embd->data(tokens[i] * d),
                                       dsize(dt_logits) * d, INFINIRT_MEMCPY_D2D, stream));
    }

    for (uint32_t i = 0; i < ntok; i++) {
        RUN_INFINI(infinirtMemcpyAsync(pos_buf->data(i * d),
                                       weight->w_pos_embd->data(pos[i] * d),
                                       dsize(dt_logits) * d, INFINIRT_MEMCPY_D2D, stream));
    }

    for (uint32_t i = 0; i < ntok; i++) {
        RUN_INFINI(infinirtMemcpyAsync(token_type_buf->data(i * d),
                                       weight->w_tok_embd->data(token_type_ids[i] * d),
                                       dsize(dt_logits) * d, INFINIRT_MEMCPY_D2D, stream));
    }
    add(logits_in, logits_in, pos_buf);
    add(logits_in, logits_in, token_type_buf);
    layerNorm(logits_in, nullptr, nullptr, logits_in, weight->w_layer_norm, weight->b_layer_norm, 1e-5);

    // 接下来就是Attention层的计算
    for (uint32_t layer = 0; layer < nlayer; layer++) {
        rearrange(logits_in_copy, logits_in); // 保存一份输入用于残差连接
        linear(q_buf, logits_in, weight->w_attn_q[layer]->view_as({d, d})->permute({1, 0}), 1.0, 0.0, nullptr, weight->b_attn_q[layer]);
        linear(k_buf, logits_in, weight->w_attn_k[layer]->view_as({d, d})->permute({1, 0}), 1.0, 0.0, nullptr, weight->b_attn_k[layer]);
        linear(v_buf, logits_in, weight->w_attn_v[layer]->view_as({d, d})->permute({1, 0}), 1.0, 0.0, nullptr, weight->b_attn_v[layer]);

        linear(qk_buf, q_buf->view_as({q_buf->numel() / d, nh, d / nh})->permute({1, 0, 2}),
               k_buf->view_as({k_buf->numel() / d, nh, d / nh})->permute({1, 2, 0}),
               1.0f / std::sqrt(static_cast<float>(dh)), 0.0f, nullptr, nullptr);

        softmax(qk_buf, qk_buf, -1);
        linear(v_buf->view_as({v_buf->numel() / d, nh, d / nh})->permute({1, 0, 2}),
               qk_buf, v_buf->view_as({v_buf->numel() / d, nh, d / nh})->permute({1, 0, 2}),
               1.0, 0.0, nullptr, nullptr);

        linear(logits_in, v_buf,
               weight->w_attn_out[layer]->view_as({d, d})->permute({1, 0}),
               1.0, 0.0, nullptr, weight->b_attn_out[layer]);
        add(logits_in, logits_in, logits_in_copy); // residual connection
        layerNorm(logits_in, nullptr, nullptr, logits_in, weight->w_attn_layer_norm[layer], weight->b_attn_layer_norm[layer], 1e-5);

        // InterMediate Layer

        linear(inter_buf, logits_in,
               weight->w_intermediate[layer]->view_as({di, d})->permute({1, 0}),
               1.0, 0.0, nullptr, weight->b_intermediate[layer]);
        gelu(inter_buf, inter_buf);

        linear(logits_in_copy, inter_buf,
               weight->w_out[layer]->view_as({d, di})->permute({1, 0}),
               1.0, 0.0, nullptr, weight->b_out[layer]);
        add(logits_in, logits_in, logits_in_copy); // residual connection
        layerNorm(logits_in, nullptr, nullptr, logits_in, weight->w_out_layer_norm[layer], weight->b_out_layer_norm[layer], 1e-5);
    }

    linear(pooler, logits_in->slice({{0, 0, 1}})->view_as({1, d}), weight->w_pooler->permute({1, 0}), 1.0, 0.0, nullptr, weight->b_pooler);
    tanh(pooler, pooler);

    linear(sparse_out, logits_in, weight->w_sparse->permute({1, 0}), 1.0, 0.0, nullptr, weight->b_sparse);
    // sparse output
    relu(sparse_out, sparse_out);
    sparse_out->debug();

    lpNorm(logits_in->slice({{0, 0, 1}}), logits_in->slice({{0, 0, 1}}), -1, 2, 1e-12);
    // dense output
    logits_in->slice({{0, 0, 1}})->debug();
    exit(0);
}

void createDeviceResource(BGEDeviceResource *rsrc, const BGEMeta *meta,
                          std::shared_ptr<BGEDeviceWeight> weights,
                          infiniDevice_t device, int idev,
                          int ndev, int dev_id,
                          infinicclComm_t comm) {
    RUN_INFINI(infinirtSetDevice(device, dev_id));
    infiniopHandle_t handle;
    infiniopCreateHandle(&handle);
    infinirtStream_t stream;
    infinirtStreamCreate(&stream);

    auto memory_pool = std::make_shared<MemoryPool>(128 * 1024 * 1024);

    *rsrc = BGEDeviceResource{
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

void releaseDeviceResource(BGEDeviceResource &res) {
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
inferBatchBGE(struct BGEModel *model,
              const uint32_t *tokens, uint32_t ntok,
              const uint32_t *req_lens, uint32_t nreq, const uint32_t *req_pos,
              struct KVCache **kv_caches,
              const float *temperature, const uint32_t *topk, const float *topp,
              uint32_t *output) {
    model->req.tokens = tokens;
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
forwardBatchBGE(struct BGEModel *model,
                const uint32_t *tokens, uint32_t ntok,
                const uint32_t *req_lens, uint32_t nreq, const uint32_t *req_pos,
                struct KVCache **kv_caches,
                void *logits) {
    model->req.tokens = tokens;
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

void launchDevice(const BGEMeta *meta, std::shared_ptr<BGEDeviceWeight> weights, BGEDeviceResource *rsrc, InferState &state, InferRequest &req,
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
        inferDeviceBatch(meta, *rsrc, idev, ndev, req.tokens, req.ntok,
                         req.req_lens, req.nreq, req.req_pos, req.output, req.logits);

        state.proceed = false;
        lock.unlock();
        state.cv_done.notify_one();
    }

    // Clean-Up
    releaseDeviceResource(*rsrc);
    setInferenceContext(nullptr); // Clear the context when done
}

BGEModel::BGEModel(const BGEMeta *meta, const ModelWeights *weights_) {
    auto weights = (BGEWeights *)(weights_);
    device = weights->device();
    dev_ids = weights->devIds();
    int ndev = int(dev_ids.size());
    dev_resources = std::vector<BGEDeviceResource>(ndev);
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

__C struct BGEModel *
createBGEModel(const BGEMeta *meta,
               const ModelWeights *weights) {
    BGEModel *model = new BGEModel(meta, weights);
    return model;
}

__C void destroyBGEModel(struct BGEModel *model) {
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