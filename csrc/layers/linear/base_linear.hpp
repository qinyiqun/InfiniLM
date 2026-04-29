#pragma once

#include "infinicore/ops.hpp"
#include "../quantization/quantization.hpp"
#include "infinicore/nn/module.hpp"
#include <infiniccl.h>
#include <optional>

namespace infinilm::nn {

using namespace infinicore::nn;

class BaseLinear : public infinicore::nn::Module {
public:
    BaseLinear(size_t in_features, size_t out_features, bool bias = true,
               const infinicore::DataType &dtype = infinicore::DataType::F32, const infinicore::Device &device = infinicore::Device());

    BaseLinear(size_t in_features, size_t out_features, std::shared_ptr<infinilm::quantization::BaseQuantization> quantization, bool bias = true,
               const infinicore::DataType &dtype = infinicore::DataType::F32, const infinicore::Device &device = infinicore::Device());

    // Forward pass: output = input @ weight.T + bias
    infinicore::Tensor forward(infinicore::Tensor &input) const;

    // Forward pass with residual connection
    infinicore::Tensor forward(infinicore::Tensor &input, infinicore::Tensor &residual) const;

    // Module information
    size_t in_features() const { return in_features_; }
    size_t out_features() const { return out_features_; }
    bool has_bias() const { return has_bias_; }
    infinicore::DataType dtype() const { return dtype_; }

    // Accessors for parameters
    infinicore::Tensor weight() const { return weight_; }
    infinicore::Tensor bias() const { return bias_; }
    infinicore::Tensor weight_scale() const { return weight_scale_; }
    infinicore::Tensor weight_zeros() const { return weight_zeros_; }
    infinicore::Tensor gidx() const { return gidx_; }

    std::shared_ptr<infinilm::quantization::BaseQuantization> get_quantization() const { return quantization_; }
    void process_weights_after_loading();

protected:
    INFINICORE_NN_PARAMETER(weight);
    INFINICORE_NN_PARAMETER(bias);

    INFINICORE_NN_PARAMETER(weight_scale);
    INFINICORE_NN_PARAMETER(weight_zeros);

    INFINICORE_NN_PARAMETER(gidx);

    infinicore::Tensor compute_linear(infinicore::Tensor &input) const;

    size_t in_features_;
    size_t out_features_;
    bool has_bias_;
    infinicore::DataType dtype_;
    std::shared_ptr<infinilm::quantization::BaseQuantization> quantization_ = std::make_shared<infinilm::quantization::NoneQuantization>(nullptr);
};

} // namespace infinilm::nn
