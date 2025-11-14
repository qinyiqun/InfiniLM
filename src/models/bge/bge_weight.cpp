#include "bge.hpp"

#include <cmath>

BGEM3Weights::BGEM3Weights(
    const BGEM3Meta *meta,
    infiniDevice_t device,
    const std::vector<int> &dev_ids) : infinicore::weights::Loader(device, dev_ids) {
    auto ndev = dev_ids.size();
    _device_weights.resize(ndev);
    infiniDtype_t dt_logits = meta->dt_logits;
    size_t nlayer = meta->nlayer;
    size_t d = meta->d;
    size_t di = meta->di / ndev;
    size_t dctx = meta->dctx;
    size_t dvoc = meta->dvoc;

    std::cout << "nlayer: " << nlayer << ", d: " << d << ", di: " << di << ", dctx: " << dctx << ", dvoc: " << dvoc << std::endl;

    for (size_t i = 0; i < ndev; i++) {
        RUN_INFINI(infinirtSetDevice(device, dev_ids[i]));

        auto weight = std::make_shared<BGEM3DeviceWeight>();
        _device_weights[i] = weight;

        auto w_word_embd = Tensor::weight(nullptr, dt_logits, {dvoc, d});
        this->register_weight("embeddings.word_embeddings.weight", w_word_embd, i);
        weight->w_word_embd = w_word_embd;

        auto w_pos_embd = Tensor::weight(nullptr, dt_logits, {dctx, d});
        this->register_weight("embeddings.position_embeddings.weight", w_pos_embd, i);
        weight->w_pos_embd = w_pos_embd;

        auto w_tok_embd = Tensor::weight(nullptr, dt_logits, {1, d});
        this->register_weight("embeddings.token_type_embeddings.weight", w_tok_embd, i);
        weight->w_tok_embd = w_tok_embd;

        auto w_layer_norm = Tensor::weight(nullptr, dt_logits, {d});
        this->register_weight("embeddings.LayerNorm.weight", w_layer_norm, i);
        weight->w_layer_norm = w_layer_norm;

        auto b_layer_norm = Tensor::weight(nullptr, dt_logits, {d});
        this->register_weight("embeddings.LayerNorm.bias", b_layer_norm, i);
        weight->b_layer_norm = b_layer_norm;

        auto w_colbert = Tensor::weight(nullptr, dt_logits, {d, d});
        this->register_weight("colbert.Linear.weight", w_colbert, i);
        weight->w_colbert = w_colbert;

        auto b_colbert = Tensor::weight(nullptr, dt_logits, {d});
        this->register_weight("colbert.Linear.bias", b_colbert, i);
        weight->b_colbert = b_colbert;

        auto w_sparse = Tensor::weight(nullptr, dt_logits, {1, d});
        this->register_weight("sparse.Linear.weight", w_sparse, i);
        weight->w_sparse = w_sparse;

        auto b_sparse = Tensor::weight(nullptr, dt_logits, {1});
        this->register_weight("sparse.Linear.bias", b_sparse, i);
        weight->b_sparse = b_sparse;

        for (size_t layer = 0; layer < nlayer; layer++) {

#define RIGISTER_LAYER_WEIGHT(W_NAME, W_VAR, W_SHAPE, W_DTYPE, W_DIST_TYPE)                      \
    auto W_VAR = Tensor::weight(nullptr, W_DTYPE, W_SHAPE);                                      \
    this->register_weight(W_NAME, W_VAR, i, infinicore::weights::DistributionType::W_DIST_TYPE); \
    weight->W_VAR.push_back(W_VAR);

            RIGISTER_LAYER_WEIGHT("encoder.layer." + std::to_string(layer) + ".attention.self.query.weight", w_attn_q, {d * d}, dt_logits, ROW);
            RIGISTER_LAYER_WEIGHT("encoder.layer." + std::to_string(layer) + ".attention.self.query.bias", b_attn_q, {d}, dt_logits, FULL);
            RIGISTER_LAYER_WEIGHT("encoder.layer." + std::to_string(layer) + ".attention.self.key.weight", w_attn_k, {d * d}, dt_logits, ROW);
            RIGISTER_LAYER_WEIGHT("encoder.layer." + std::to_string(layer) + ".attention.self.key.bias", b_attn_k, {d}, dt_logits, FULL);
            RIGISTER_LAYER_WEIGHT("encoder.layer." + std::to_string(layer) + ".attention.self.value.weight", w_attn_v, {d * d}, dt_logits, ROW);
            RIGISTER_LAYER_WEIGHT("encoder.layer." + std::to_string(layer) + ".attention.self.value.bias", b_attn_v, {d}, dt_logits, FULL);

            RIGISTER_LAYER_WEIGHT("encoder.layer." + std::to_string(layer) + ".attention.output.dense.weight", w_attn_out, {d * d}, dt_logits, ROW);
            RIGISTER_LAYER_WEIGHT("encoder.layer." + std::to_string(layer) + ".attention.output.dense.bias", b_attn_out, {d}, dt_logits, ROW);

            RIGISTER_LAYER_WEIGHT("encoder.layer." + std::to_string(layer) + ".attention.output.LayerNorm.weight", w_attn_layer_norm, {d}, dt_logits, ROW);
            RIGISTER_LAYER_WEIGHT("encoder.layer." + std::to_string(layer) + ".attention.output.LayerNorm.bias", b_attn_layer_norm, {d}, dt_logits, ROW);

            RIGISTER_LAYER_WEIGHT("encoder.layer." + std::to_string(layer) + ".intermediate.dense.weight", w_intermediate, {d * di}, dt_logits, ROW);
            RIGISTER_LAYER_WEIGHT("encoder.layer." + std::to_string(layer) + ".intermediate.dense.bias", b_intermediate, {di}, dt_logits, FULL);

            RIGISTER_LAYER_WEIGHT("encoder.layer." + std::to_string(layer) + ".output.dense.weight", w_out, {di * d}, dt_logits, ROW);
            RIGISTER_LAYER_WEIGHT("encoder.layer." + std::to_string(layer) + ".output.dense.bias", b_out, {d}, dt_logits, FULL);
            RIGISTER_LAYER_WEIGHT("encoder.layer." + std::to_string(layer) + ".output.LayerNorm.weight", w_out_layer_norm, {d}, dt_logits, FULL);
            RIGISTER_LAYER_WEIGHT("encoder.layer." + std::to_string(layer) + ".output.LayerNorm.bias", b_out_layer_norm, {d}, dt_logits, FULL);
        }

        auto w_pooler = Tensor::weight(nullptr, dt_logits, {d, d});
        this->register_weight("pooler.dense.weight", w_pooler, i);
        weight->w_pooler = w_pooler;

        auto b_pooler = Tensor::weight(nullptr, dt_logits, {d});
        this->register_weight("pooler.dense.bias", b_pooler, i);
        weight->b_pooler = b_pooler;
    }

#undef RIGISTER_LAYER_WEIGHT
}

__C struct ModelWeights *
createBGEM3Weights(const BGEM3Meta *meta,
                   infiniDevice_t device,
                   int ndev,
                   const int *dev_ids) {
    BGEM3Weights *weights = new BGEM3Weights(meta, device, std::vector<int>(dev_ids, dev_ids + ndev));
    return (struct ModelWeights *)weights;
}