#include "linear.hpp"
#include "infinicore/ops.hpp"
#include "infinicore/ops/distributed/allreduce.hpp"
#include "infinicore/ops/linear.hpp"
#include <optional>
#include <spdlog/spdlog.h>

namespace infinilm::nn {

Linear::Linear(size_t in_features, size_t out_features, bool bias,
               const infinicore::DataType &dtype, const infinicore::Device &device)
    : BaseLinear(in_features, out_features, bias, dtype, device_) {

    device_ = device;

    INFINICORE_NN_PARAMETER_INIT(weight, ({out_features, in_features}, dtype_, device));

    if (bias) {
        INFINICORE_NN_PARAMETER_INIT(bias, ({out_features}, dtype_, device));
    } else {
        bias_ = infinicore::nn::Parameter();
    }
}

Linear::Linear(size_t in_features, size_t out_features,
               std::shared_ptr<infinilm::quantization::BaseQuantization> quantization, bool bias,
               const infinicore::DataType &dtype, const infinicore::Device &device)
    : BaseLinear(in_features, out_features, quantization, bias, dtype, device_) {

    device_ = device;

    switch (this->quantization_->get_quant_scheme()) {
    case infinilm::quantization::QuantScheme::COMPRESSED_TENSOR_W8A8I8: {
        INFINICORE_NN_PARAMETER_INIT(weight, ({out_features, in_features}, infinicore::DataType::I8, device));
        INFINICORE_NN_PARAMETER_INIT(weight_scale, ({out_features, 1}, infinicore::DataType::F32, device));

        if (bias) {
            INFINICORE_NN_PARAMETER_INIT(bias, ({out_features}, dtype_, device));
        } else {
            bias_ = infinicore::nn::Parameter();
        }
        break;
    }
    case infinilm::quantization::QuantScheme::AWQ_W4A16: {
        weight_ = infinicore::nn::Parameter({out_features, in_features}, infinicore::DataType::I32, device);
        this->register_parameter("qweight", weight_);
        weight_zeros_ = infinicore::nn::Parameter({out_features, in_features}, infinicore::DataType::I32, device);
        this->register_parameter("qzeros", weight_zeros_);
        weight_scale_ = infinicore::nn::Parameter({out_features, in_features}, dtype_, device);
        this->register_parameter("scales", weight_scale_);
        if (bias) {
            INFINICORE_NN_PARAMETER_INIT(bias, ({out_features}, dtype_, device));
        } else {
            bias_ = infinicore::nn::Parameter();
        }
        break;
    }
    case infinilm::quantization::QuantScheme::GPTQ_W4A16_QY: {
        weight_ = infinicore::nn::Parameter({in_features / 2, out_features}, infinicore::DataType::U8, device);
        this->register_parameter("qweight", weight_);
        weight_zeros_ = infinicore::nn::Parameter({in_features / 128, out_features}, dtype_, device);
        this->register_parameter("qzeros", weight_zeros_);
        weight_scale_ = infinicore::nn::Parameter({in_features / 128, out_features}, dtype_, device);
        this->register_parameter("scales", weight_scale_);

        gidx_ = infinicore::nn::Parameter({in_features}, infinicore::DataType::I32, device);
        this->register_parameter("g_idx", gidx_);
        if (bias) {
            INFINICORE_NN_PARAMETER_INIT(bias, ({out_features}, dtype_, device));
        } else {
            bias_ = infinicore::nn::Parameter();
        }
        break;
    }
    case infinilm::quantization::QuantScheme::GPTQ_W4A16: {
        weight_ = infinicore::nn::Parameter({in_features / 8, out_features}, infinicore::DataType::I32, device);
        this->register_parameter("qweight", weight_);
        weight_zeros_ = infinicore::nn::Parameter({in_features / 128, out_features / 8}, infinicore::DataType::I32, device);
        this->register_parameter("qzeros", weight_zeros_);
        weight_scale_ = infinicore::nn::Parameter({in_features / 128, out_features}, dtype_, device);
        this->register_parameter("scales", weight_scale_);
        gidx_ = infinicore::nn::Parameter({in_features}, infinicore::DataType::I32, device);
        this->register_parameter("g_idx", gidx_);
        if (bias) {
            INFINICORE_NN_PARAMETER_INIT(bias, ({out_features}, dtype_, device));
        } else {
            bias_ = infinicore::nn::Parameter();
        }
        break;
    }
    default: {
        INFINICORE_NN_PARAMETER_INIT(weight, ({out_features, in_features}, dtype_, device));

        if (bias) {
            INFINICORE_NN_PARAMETER_INIT(bias, ({out_features}, dtype_, device));
        } else {
            bias_ = infinicore::nn::Parameter();
        }
        break;
    }
    }
}

infinicore::Tensor Linear::forward(infinicore::Tensor &input) const {
    return BaseLinear::forward(input);
}

std::string Linear::extra_repr() const {
    return "Linear(in_features=" + std::to_string(in_features_) + ", out_features=" + std::to_string(out_features_) + ", bias=" + (has_bias_ ? "true" : "false") + ", dtype=" + std::to_string(static_cast<int>(dtype_)) + ")";
}

} // namespace infinilm::nn

namespace infinilm::nn {

ColumnParallelLinear::ColumnParallelLinear(size_t in_features, size_t out_features, bool bias,
                                           const infinicore::DataType &dtype, const infinicore::Device &device,
                                           infinicore::Size tp_rank, infinicore::Size tp_size)
    : BaseLinear(in_features, out_features, bias, dtype, device_),
      tp_rank_(tp_rank),
      tp_size_(tp_size) {

    device_ = device;

    INFINICORE_NN_PARAMETER_INIT(weight, ({out_features, in_features}, dtype_, device,
                                          0, tp_rank_, tp_size_));

    if (bias) {
        INFINICORE_NN_PARAMETER_INIT(bias, ({out_features}, dtype_, device,
                                            0, tp_rank_, tp_size_));
    } else {
        bias_ = infinicore::nn::Parameter();
    }
}

ColumnParallelLinear::ColumnParallelLinear(size_t in_features, size_t out_features, std::shared_ptr<infinilm::quantization::BaseQuantization> quantization, bool bias,
                                           const infinicore::DataType &dtype, const infinicore::Device &device,
                                           infinicore::Size tp_rank, infinicore::Size tp_size)
    : BaseLinear(in_features, out_features, quantization, bias, dtype, device_),
      tp_rank_(tp_rank),
      tp_size_(tp_size) {

    device_ = device;

    switch (this->quantization_->get_quant_scheme()) {
    case infinilm::quantization::QuantScheme::COMPRESSED_TENSOR_W8A8I8: {

        INFINICORE_NN_PARAMETER_INIT(weight, ({out_features, in_features}, infinicore::DataType::I8, device, 0, tp_rank_, tp_size_));
        INFINICORE_NN_PARAMETER_INIT(weight_scale, ({out_features, 1}, infinicore::DataType::F32, device, 0, tp_rank_, tp_size_));

        if (bias) {
            INFINICORE_NN_PARAMETER_INIT(bias, ({out_features}, dtype_, device, 0, 0, 1));
        } else {
            bias_ = infinicore::nn::Parameter();
        }
        break;
    }
    case infinilm::quantization::QuantScheme::AWQ_W4A16: {
        auto awq_ptr = std::static_pointer_cast<infinilm::quantization::AWQ>(this->quantization_);
        int group_size = awq_ptr->get_group_size();
        int packing_num = awq_ptr->get_packing_num();

        weight_ = infinicore::nn::Parameter({in_features, out_features / packing_num},
                                            infinicore::DataType::I32,
                                            device, 1, tp_rank_, tp_size_);
        this->register_parameter("qweight", weight_);

        weight_scale_ = infinicore::nn::Parameter({in_features / group_size, out_features},
                                                  dtype_,
                                                  device, 1, tp_rank_, tp_size_);
        this->register_parameter("scales", weight_scale_);

        weight_zeros_ = infinicore::nn::Parameter({in_features / group_size, out_features / packing_num},
                                                  infinicore::DataType::I32,
                                                  device, 1, tp_rank_, tp_size_);

        this->register_parameter("qzeros", weight_zeros_);
        if (bias) {
            INFINICORE_NN_PARAMETER_INIT(bias, ({out_features}, dtype_, device, 0, 0, 1));
        } else {
            bias_ = infinicore::nn::Parameter();
        }
        break;
    }
    case infinilm::quantization::QuantScheme::GPTQ_W4A16_QY: {
        auto gptq_ptr = std::static_pointer_cast<infinilm::quantization::GPTQ_QY>(this->quantization_);
        int group_size = gptq_ptr->get_group_size();
        int packing_num = gptq_ptr->get_packing_num();
        weight_ = infinicore::nn::Parameter({in_features / 2, out_features}, infinicore::DataType::U8, device, 1, tp_rank_, tp_size_);
        this->register_parameter("qweight", weight_);
        weight_zeros_ = infinicore::nn::Parameter({in_features / group_size, out_features}, dtype_, device, 1, tp_rank_, tp_size_);
        this->register_parameter("qzeros", weight_zeros_);
        weight_scale_ = infinicore::nn::Parameter({in_features / group_size, out_features}, dtype_, device, 1, tp_rank_, tp_size_);
        this->register_parameter("scales", weight_scale_);
        gidx_ = infinicore::nn::Parameter({in_features},
                                          infinicore::DataType::I32,
                                          device, 0, tp_rank_, tp_size_);
        this->register_parameter("g_idx", gidx_);
        if (bias) {
            INFINICORE_NN_PARAMETER_INIT(bias, ({out_features}, dtype_, device, 0, tp_rank_, tp_size_));
        } else {
            bias_ = infinicore::nn::Parameter();
        }
        break;
    }
    case infinilm::quantization::QuantScheme::GPTQ_W4A16: {
        auto gptq_ptr = std::static_pointer_cast<infinilm::quantization::GPTQ>(this->quantization_);
        int group_size = gptq_ptr->get_group_size();
        int packing_num = gptq_ptr->get_packing_num();
        weight_ = infinicore::nn::Parameter({in_features / 8, out_features}, infinicore::DataType::I32, device, 1, tp_rank_, tp_size_);
        this->register_parameter("qweight", weight_);
        weight_zeros_ = infinicore::nn::Parameter({in_features / group_size, out_features / 8}, infinicore::DataType::I32, device, 1, tp_rank_, tp_size_);
        this->register_parameter("qzeros", weight_zeros_);
        weight_scale_ = infinicore::nn::Parameter({in_features / group_size, out_features}, dtype_, device, 1, tp_rank_, tp_size_);
        this->register_parameter("scales", weight_scale_);
        gidx_ = infinicore::nn::Parameter({in_features},
                                          infinicore::DataType::I32,
                                          device, 0, tp_rank_, tp_size_);
        this->register_parameter("g_idx", gidx_);
        if (bias) {
            INFINICORE_NN_PARAMETER_INIT(bias, ({out_features}, dtype_, device, 0, tp_rank_, tp_size_));
        } else {
            bias_ = infinicore::nn::Parameter();
        }
        break;
    }
    default: {
        INFINICORE_NN_PARAMETER_INIT(weight, ({out_features, in_features}, dtype_, device,
                                              0, tp_rank_, tp_size_));

        if (bias) {
            INFINICORE_NN_PARAMETER_INIT(bias, ({out_features}, dtype_, device,
                                                0, tp_rank_, tp_size_));
        } else {
            bias_ = infinicore::nn::Parameter();
        }
        break;
    }
    }
}

infinicore::Tensor ColumnParallelLinear::forward(infinicore::Tensor &input) const {
    return BaseLinear::forward(input);
}

std::string ColumnParallelLinear::extra_repr() const {
    return "ColumnParallelLinear(in_features=" + std::to_string(in_features_) + ", out_features=" + std::to_string(out_features_) + ", bias=" + (has_bias_ ? "true" : "false") + ", dtype=" + std::to_string(static_cast<int>(dtype_)) + ")";
}

} // namespace infinilm::nn

namespace infinilm::nn {

RowParallelLinear::RowParallelLinear(size_t in_features, size_t out_features, bool bias,
                                     const infinicore::DataType &dtype, const infinicore::Device &device,
                                     infinicore::Size tp_rank, infinicore::Size tp_size, infinicclComm_t communicator)
    : BaseLinear(in_features, out_features, bias, dtype, device_),
      tp_rank_(tp_rank),
      tp_size_(tp_size), communicator_(communicator) {

    device_ = device;

    INFINICORE_NN_PARAMETER_INIT(weight, ({out_features, in_features}, dtype_, device,
                                          1, tp_rank_, tp_size_));

    if (bias && (0 == tp_rank_)) {
        INFINICORE_NN_PARAMETER_INIT(bias, ({out_features}, dtype_, device, 0, 0, 1));
    } else {
        bias_ = infinicore::nn::Parameter();
    }
}

RowParallelLinear::RowParallelLinear(size_t in_features, size_t out_features, std::shared_ptr<infinilm::quantization::BaseQuantization> quantization, bool bias,
                                     const infinicore::DataType &dtype, const infinicore::Device &device,
                                     infinicore::Size tp_rank, infinicore::Size tp_size, infinicclComm_t communicator)
    : BaseLinear(in_features, out_features, quantization, bias, dtype, device_),
      tp_rank_(tp_rank),
      tp_size_(tp_size), communicator_(communicator) {

    device_ = device;

    switch (this->quantization_->get_quant_scheme()) {
    case infinilm::quantization::QuantScheme::COMPRESSED_TENSOR_W8A8I8: {
        INFINICORE_NN_PARAMETER_INIT(weight, ({out_features, in_features}, infinicore::DataType::I8, device, 1, tp_rank_, tp_size_));
        INFINICORE_NN_PARAMETER_INIT(weight_scale, ({out_features, 1}, infinicore::DataType::F32, device, 0, 0, 1));

        if (bias) {
            INFINICORE_NN_PARAMETER_INIT(bias, ({out_features}, dtype_, device, 0, tp_rank_, tp_size_));
        } else {
            bias_ = infinicore::nn::Parameter();
        }
        break;
    }
    case infinilm::quantization::QuantScheme::AWQ_W4A16: {
        auto awq_ptr = std::static_pointer_cast<infinilm::quantization::AWQ>(this->quantization_);
        int group_size = awq_ptr->get_group_size();
        int packing_num = awq_ptr->get_packing_num();

        weight_ = infinicore::nn::Parameter({in_features, out_features / packing_num},
                                            infinicore::DataType::I32,
                                            device, 0, tp_rank_, tp_size_);
        this->register_parameter("qweight", weight_);

        weight_scale_ = infinicore::nn::Parameter({in_features / group_size, out_features},
                                                  dtype_,
                                                  device, 0, tp_rank_, tp_size_);
        this->register_parameter("scales", weight_scale_);
        weight_zeros_ = infinicore::nn::Parameter({in_features / group_size, out_features / packing_num},
                                                  infinicore::DataType::I32,
                                                  device, 0, tp_rank_, tp_size_);
        this->register_parameter("qzeros", weight_zeros_);

        if (bias && (0 == tp_rank_)) {
            INFINICORE_NN_PARAMETER_INIT(bias, ({out_features}, dtype_, device, 0, 0, 1));
        } else {
            bias_ = infinicore::nn::Parameter();
        }
        break;
    }
    case infinilm::quantization::QuantScheme::GPTQ_W4A16_QY: {
        auto gptq_ptr = std::static_pointer_cast<infinilm::quantization::GPTQ_QY>(this->quantization_);
        int group_size = gptq_ptr->get_group_size();
        int packing_num = gptq_ptr->get_packing_num();

        weight_ = infinicore::nn::Parameter({in_features / 2, out_features}, infinicore::DataType::U8, device, 0, tp_rank_, tp_size_);
        this->register_parameter("qweight", weight_);
        weight_zeros_ = infinicore::nn::Parameter({in_features / group_size, out_features}, dtype_, device, 0, tp_rank_, tp_size_);
        this->register_parameter("qzeros", weight_zeros_);
        weight_scale_ = infinicore::nn::Parameter({in_features / group_size, out_features}, dtype_, device, 0, tp_rank_, tp_size_);
        this->register_parameter("scales", weight_scale_);

        gidx_ = infinicore::nn::Parameter({in_features},
                                          infinicore::DataType::I32,
                                          device, 0, tp_rank_, tp_size_);
        this->register_parameter("g_idx", gidx_);
        if (bias && (0 == tp_rank_)) {
            INFINICORE_NN_PARAMETER_INIT(bias, ({out_features}, dtype_, device, 0, 0, 1));
        } else {
            bias_ = infinicore::nn::Parameter();
        }
        break;
    }
    case infinilm::quantization::QuantScheme::GPTQ_W4A16: {
        auto gptq_ptr = std::static_pointer_cast<infinilm::quantization::GPTQ>(this->quantization_);
        int group_size = gptq_ptr->get_group_size();
        int packing_num = gptq_ptr->get_packing_num();

        weight_ = infinicore::nn::Parameter({in_features / 8, out_features}, infinicore::DataType::I32, device, 0, tp_rank_, tp_size_);
        this->register_parameter("qweight", weight_);
        weight_zeros_ = infinicore::nn::Parameter({in_features / group_size, out_features / 8}, infinicore::DataType::I32, device, 0, tp_rank_, tp_size_);
        this->register_parameter("qzeros", weight_zeros_);
        weight_scale_ = infinicore::nn::Parameter({in_features / group_size, out_features}, dtype_, device, 0, tp_rank_, tp_size_);
        this->register_parameter("scales", weight_scale_);

        gidx_ = infinicore::nn::Parameter({in_features},
                                          infinicore::DataType::I32,
                                          device, 0, tp_rank_, tp_size_);
        this->register_parameter("g_idx", gidx_);
        if (bias && (0 == tp_rank_)) {
            INFINICORE_NN_PARAMETER_INIT(bias, ({out_features}, dtype_, device, 0, 0, 1));
        } else {
            bias_ = infinicore::nn::Parameter();
        }
        break;
    }
    default: {
        INFINICORE_NN_PARAMETER_INIT(weight, ({out_features, in_features}, dtype_, device,
                                              1, tp_rank_, tp_size_));

        if (bias && (0 == tp_rank_)) {
            INFINICORE_NN_PARAMETER_INIT(bias, ({out_features}, dtype_, device, 0, 0, 1));
        } else {
            bias_ = infinicore::nn::Parameter();
        }
        break;
    }
    }
}

infinicore::Tensor RowParallelLinear::forward(infinicore::Tensor &input) const {
    auto output = BaseLinear::forward(input);

    if ((tp_size_ > 1) && (communicator_ != nullptr)) {
        infinicore::op::distributed::allreduce_(output, output, INFINICCL_SUM, communicator_);
    }
    return output;
}

std::string RowParallelLinear::extra_repr() const {
    return "RowParallelLinear(in_features=" + std::to_string(in_features_) + ", out_features=" + std::to_string(out_features_) + ", bias=" + (has_bias_ ? "true" : "false") + ", dtype=" + std::to_string(static_cast<int>(dtype_)) + ")";
}

} // namespace infinilm::nn
