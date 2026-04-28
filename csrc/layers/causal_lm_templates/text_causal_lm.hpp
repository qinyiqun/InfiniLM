#pragma once

#include "../../models/infinilm_model.hpp"
#include "../attention/attention.hpp"
#include "../linear/linear.hpp"
#include "../mlp/mlp.hpp"
#include "infinicore/device.hpp"
#include "iostream"

namespace infinilm::layers::causal_lm_templates {

/**
 * @brief Text Causal Language Modeling class
 *
 * A generic template class for Causal Language Modeling.
 *
 * @tparam Model The base model type (e.g., Qwen3Model, Qwen3MoeModel)
 *
 * Usage example:
 * @code
 * using Qwen3CausalLM = TextCausalLM<Qwen3Model>;
 * @endcode
 */
template <typename Model>
class TextCausalLM : public InfinilmModel {
public:
    /**
     * @brief Construct TextCausalLM module
     *
     * @param model_config: Model configuration.
     * @param device: Device to create tensors on
     */
    TextCausalLM(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                 const infinicore::Device &device) {
        model_config_ = model_config;

        size_t hidden_size = model_config->get<size_t>("hidden_size");
        size_t vocab_size = model_config->get<size_t>("vocab_size");
        const auto &dtype{model_config->get_dtype()};

        model_ = this->register_module<Model>("model", model_config, device);
        lm_head_ = this->register_module<infinilm::layers::linear::ReplicatedLinear>("lm_head", hidden_size, vocab_size, false, dtype, device);
    }

    /**
     * @brief Forward pass: compute language modeling logits
     */
    Output forward(const Input &input) const override {
        auto hidden_states = model_->forward(input);
        auto logits = lm_head_->forward(hidden_states);
        return {logits};
    }

    inline bool is_module_type(const std::string &type_name, const std::string &target) {
        return type_name.find(target) != std::string::npos;
    }

    void process_weights_after_loading() const override {
        // ✅ 现在可以正常编译
        auto modules = model_->modules_dict();

        for (const auto &[name, mod] : modules) {

            if (name.find("self_attn") != std::string::npos) {
                if (auto *attn = dynamic_cast<infinilm::layers::attention::Attention *>(mod)) {
                    // attn->get_attention_qkv_weight_info();
                    attn->process_weights_after_loading();
                }
            }
            // 🔹 匹配 MLP 模块（名字以 .mlp 结尾或是根节点 mlp）
            else if ((name.find(".mlp") != std::string::npos && name.find('.', name.find(".mlp") + 4) == std::string::npos) || name == "mlp") {
                if (auto *mlp = dynamic_cast<infinilm::layers::mlp::MLP *>(mod)) {
                    mlp->process_weights_after_loading();
                }
            }
        }
    }

protected:
    INFINICORE_NN_MODULE(Model, model);
    INFINICORE_NN_MODULE(infinilm::layers::linear::ReplicatedLinear, lm_head);
};

} // namespace infinilm::layers::causal_lm_templates
