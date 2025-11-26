#include "bge_reranker.hpp"

#include <cmath>

BGERerankerWeights::BGERerankerWeights(
    const BGERerankerMeta *meta,
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

        auto weight = std::make_shared<BGERerankerDeviceWeight>();
        _device_weights[i] = weight;

        auto w_word_embd = Tensor::weight(nullptr, dt_logits, {dvoc, d});
        this->register_weight("roberta.embeddings.word_embeddings.weight", w_word_embd, i);
        weight->w_word_embd = w_word_embd;

        auto w_pos_embd = Tensor::weight(nullptr, dt_logits, {dctx, d});
        this->register_weight("roberta.embeddings.position_embeddings.weight", w_pos_embd, i);
        weight->w_pos_embd = w_pos_embd;

        auto w_tok_embd = Tensor::weight(nullptr, dt_logits, {1, d});
        this->register_weight("roberta.embeddings.token_type_embeddings.weight", w_tok_embd, i);
        weight->w_tok_embd = w_tok_embd;

        auto w_layer_norm = Tensor::weight(nullptr, dt_logits, {d});
        this->register_weight("roberta.embeddings.LayerNorm.weight", w_layer_norm, i);
        weight->w_layer_norm = w_layer_norm;

        auto b_layer_norm = Tensor::weight(nullptr, dt_logits, {d});
        this->register_weight("roberta.embeddings.LayerNorm.bias", b_layer_norm, i);
        weight->b_layer_norm = b_layer_norm;

        for (size_t layer = 0; layer < nlayer; layer++) {

#define RIGISTER_LAYER_WEIGHT(W_NAME, W_VAR, W_SHAPE, W_DTYPE, W_DIST_TYPE)                      \
    auto W_VAR = Tensor::weight(nullptr, W_DTYPE, W_SHAPE);                                      \
    this->register_weight(W_NAME, W_VAR, i, infinicore::weights::DistributionType::W_DIST_TYPE); \
    weight->W_VAR.push_back(W_VAR);

            RIGISTER_LAYER_WEIGHT("roberta.encoder.layer." + std::to_string(layer) + ".attention.self.query.weight", w_attn_q, {d * d}, dt_logits, ROW);
            RIGISTER_LAYER_WEIGHT("roberta.encoder.layer." + std::to_string(layer) + ".attention.self.query.bias", b_attn_q, {d}, dt_logits, FULL);
            RIGISTER_LAYER_WEIGHT("roberta.encoder.layer." + std::to_string(layer) + ".attention.self.key.weight", w_attn_k, {d * d}, dt_logits, ROW);
            RIGISTER_LAYER_WEIGHT("roberta.encoder.layer." + std::to_string(layer) + ".attention.self.key.bias", b_attn_k, {d}, dt_logits, FULL);
            RIGISTER_LAYER_WEIGHT("roberta.encoder.layer." + std::to_string(layer) + ".attention.self.value.weight", w_attn_v, {d * d}, dt_logits, ROW);
            RIGISTER_LAYER_WEIGHT("roberta.encoder.layer." + std::to_string(layer) + ".attention.self.value.bias", b_attn_v, {d}, dt_logits, FULL);

            RIGISTER_LAYER_WEIGHT("roberta.encoder.layer." + std::to_string(layer) + ".attention.output.dense.weight", w_attn_out, {d * d}, dt_logits, ROW);
            RIGISTER_LAYER_WEIGHT("roberta.encoder.layer." + std::to_string(layer) + ".attention.output.dense.bias", b_attn_out, {d}, dt_logits, ROW);

            RIGISTER_LAYER_WEIGHT("roberta.encoder.layer." + std::to_string(layer) + ".attention.output.LayerNorm.weight", w_attn_layer_norm, {d}, dt_logits, ROW);
            RIGISTER_LAYER_WEIGHT("roberta.encoder.layer." + std::to_string(layer) + ".attention.output.LayerNorm.bias", b_attn_layer_norm, {d}, dt_logits, ROW);

            RIGISTER_LAYER_WEIGHT("roberta.encoder.layer." + std::to_string(layer) + ".intermediate.dense.weight", w_intermediate, {d * di}, dt_logits, ROW);
            RIGISTER_LAYER_WEIGHT("roberta.encoder.layer." + std::to_string(layer) + ".intermediate.dense.bias", b_intermediate, {di}, dt_logits, FULL);

            RIGISTER_LAYER_WEIGHT("roberta.encoder.layer." + std::to_string(layer) + ".output.dense.weight", w_out, {di * d}, dt_logits, ROW);
            RIGISTER_LAYER_WEIGHT("roberta.encoder.layer." + std::to_string(layer) + ".output.dense.bias", b_out, {d}, dt_logits, FULL);
            RIGISTER_LAYER_WEIGHT("roberta.encoder.layer." + std::to_string(layer) + ".output.LayerNorm.weight", w_out_layer_norm, {d}, dt_logits, FULL);
            RIGISTER_LAYER_WEIGHT("roberta.encoder.layer." + std::to_string(layer) + ".output.LayerNorm.bias", b_out_layer_norm, {d}, dt_logits, FULL);
        }

        // w_dense_cls, b_dense_cls, w_out_cls, b_out_cls

        auto w_dense_cls = Tensor::weight(nullptr, dt_logits, {d, d});
        this->register_weight("classifier.dense.weight", w_dense_cls, i);
        weight->w_dense_cls = w_dense_cls;

        auto b_dense_cls = Tensor::weight(nullptr, dt_logits, {d});
        this->register_weight("classifier.dense.bias", b_dense_cls, i);
        weight->b_dense_cls = b_dense_cls;

        auto w_out_cls = Tensor::weight(nullptr, dt_logits, {1, d});
        this->register_weight("classifier.out_proj.weight", w_out_cls, i);
        weight->w_out_cls = w_out_cls;

        auto b_out_cls = Tensor::weight(nullptr, dt_logits, {1});
        this->register_weight("classifier.out_proj.bias", b_out_cls, i);
        weight->b_out_cls = b_out_cls;
    }

#undef RIGISTER_LAYER_WEIGHT
}

__C struct ModelWeights *
createBGERerankerWeights(const BGERerankerMeta *meta,
                         infiniDevice_t device,
                         int ndev,
                         const int *dev_ids) {
    BGERerankerWeights *weights = new BGERerankerWeights(meta, device, std::vector<int>(dev_ids, dev_ids + ndev));
    return (struct ModelWeights *)weights;
}