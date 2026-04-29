#pragma once
#include "base_quantization.hpp"
namespace infinilm::quantization {

class AWQ : public BaseQuantization {
public:
    explicit AWQ(const nlohmann::json &quant_config)
        : BaseQuantization(quant_config){};

    QuantScheme get_quant_scheme() const override {
        return QuantScheme::AWQ_W4A16;
    };

    int get_packing_num() const {
        return 32 / get_or<int>("bits", 4);
    }

    int get_group_size() const {
        return get_or<int>("group_size", 128);
    }

    std::vector<ParamDescriptor> get_param_layout(
        size_t in_features, size_t out_features,
        int split_dim, int tp_rank, int tp_size,
        int tp_num_heads,
        const infinicore::DataType &dtype,
        bool bias) const override;

    infinicore::Tensor forward(
        const ParamsMap &params,
        const infinicore::Tensor &input,
        bool has_bias) const override;

    std::vector<SplitParam> split_params(
        const std::unordered_map<std::string, infinicore::nn::Parameter> &params,
        const std::vector<SplitInfo> &splits,
        int narrow_dim,
        int tp_rank, int tp_size, int tp_num_heads) const override;
};

} // namespace infinilm::quantization
