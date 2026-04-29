#include "base_linear.hpp"
#include "infinicore/ops.hpp"
#include <spdlog/spdlog.h>

namespace infinilm::nn {

BaseLinear::BaseLinear(size_t in_features, size_t out_features,
                       std::shared_ptr<infinilm::quantization::BaseQuantization> quantization,
                       bool bias,
                       const infinicore::DataType &dtype, const infinicore::Device &device,
                       int split_dim, int tp_rank, int tp_size,
                       int tp_num_heads)
    : in_features_(in_features),
      out_features_(out_features),
      has_bias_(bias),
      dtype_(dtype),
      split_dim_(split_dim),
      quantization_(quantization) {

    device_ = device;

    auto layout = quantization_->get_param_layout(
        in_features, out_features, split_dim, tp_rank, tp_size,
        tp_num_heads, dtype, bias);

    for (const auto &desc : layout) {
        infinicore::nn::Parameter param(
            desc.shape, desc.dtype, device,
            desc.split_dim, desc.tp_rank, desc.tp_size,
            desc.tp_num_heads >= 0 ? desc.tp_num_heads : 0);
        this->register_parameter(desc.name, param);
    }
}

infinicore::Tensor BaseLinear::compute_linear(infinicore::Tensor &input) const {
    // Build params map from registered parameters
    infinilm::quantization::ParamsMap params;
    for (const auto &[name, param] : this->state_dict()) {
        params[name] = static_cast<const infinicore::Tensor &>(param);
    }

    return quantization_->forward(params, input, has_bias_);
}

infinicore::Tensor BaseLinear::forward(infinicore::Tensor &input) const {
    return compute_linear(input);
}

infinicore::Tensor BaseLinear::forward(infinicore::Tensor &input, infinicore::Tensor &residual) const {
    auto output = compute_linear(input);
    infinicore::op::add_(output, output, residual);
    return output;
}

void BaseLinear::process_weights_after_loading() {
    infinilm::quantization::ParamsMap params;
    for (const auto &[name, param] : this->state_dict()) {
        params[name] = static_cast<const infinicore::Tensor &>(param);
    }

    quantization_->process_weights_after_loading(params, device_);

    // Write back modified parameters
    for (const auto &[name, tensor] : params) {
        if (this->state_dict().count(name)) {
            this->load_parameter_(name, tensor);
        }
    }
}

// Backward compatible accessors

infinicore::Tensor BaseLinear::weight() const {
    auto &sd = this->state_dict();
    if (sd.count("weight")) return sd.at("weight");
    if (sd.count("qweight")) return sd.at("qweight");
    return infinicore::Tensor();
}

infinicore::Tensor BaseLinear::bias() const {
    auto &sd = this->state_dict();
    if (sd.count("bias")) return sd.at("bias");
    return infinicore::Tensor();
}

infinicore::Tensor BaseLinear::weight_scale() const {
    auto &sd = this->state_dict();
    if (sd.count("weight_scale")) return sd.at("weight_scale");
    if (sd.count("scales")) return sd.at("scales");
    return infinicore::Tensor();
}

infinicore::Tensor BaseLinear::weight_zeros() const {
    auto &sd = this->state_dict();
    if (sd.count("weight_zeros")) return sd.at("weight_zeros");
    if (sd.count("qzeros")) return sd.at("qzeros");
    return infinicore::Tensor();
}

infinicore::Tensor BaseLinear::gidx() const {
    auto &sd = this->state_dict();
    if (sd.count("g_idx")) return sd.at("g_idx");
    return infinicore::Tensor();
}

infinicore::Tensor BaseLinear::get_param(const std::string &name) const {
    auto &sd = this->state_dict();
    if (sd.count(name)) return sd.at(name);
    return infinicore::Tensor();
}

const infinicore::nn::Parameter &BaseLinear::get_parameter_ref(const std::string &name) const {
    return parameters_.at(name);
}

std::vector<infinilm::quantization::SplitParam> BaseLinear::split_params(
    const std::vector<infinilm::quantization::SplitInfo> &splits,
    int tp_rank, int tp_size, int tp_num_heads) const {
    return quantization_->split_params(
        parameters_, splits, split_dim_, tp_rank, tp_size, tp_num_heads);
}

} // namespace infinilm::nn
