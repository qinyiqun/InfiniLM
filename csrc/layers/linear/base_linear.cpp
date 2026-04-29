#include "base_linear.hpp"
#include "infinicore/ops.hpp"
#include "infinicore/ops/linear.hpp"
#include "infinicore/ops/linear_w4a16_awq.hpp"
#include "infinicore/ops/linear_w4a16_gptq_qy.hpp"
#include "infinicore/ops/linear_w8a8i8.hpp"
#include <optional>
#include <spdlog/spdlog.h>

namespace infinilm::nn {

BaseLinear::BaseLinear(size_t in_features, size_t out_features, bool bias,
                       const infinicore::DataType &dtype, const infinicore::Device &device)
    : in_features_(in_features),
      out_features_(out_features),
      has_bias_(bias),
      dtype_(dtype) {

    device_ = device;
}

BaseLinear::BaseLinear(size_t in_features, size_t out_features, std::shared_ptr<infinilm::quantization::BaseQuantization> quantization, bool bias,
                       const infinicore::DataType &dtype, const infinicore::Device &device)
    : in_features_(in_features),
      out_features_(out_features),
      quantization_(quantization),
      has_bias_(bias),
      dtype_(dtype) {

    device_ = device;
}

infinicore::Tensor BaseLinear::compute_linear(infinicore::Tensor &input) const {
    switch (this->quantization_->get_quant_scheme()) {
    case infinilm::quantization::QuantScheme::COMPRESSED_TENSOR_W8A8I8: {
        infinicore::Tensor input_contiguous = input->is_contiguous() ? input : input->contiguous();

        infinicore::Tensor weight_packed_tensor = static_cast<const infinicore::Tensor &>(weight_);
        infinicore::Tensor weight_scale_tensor = static_cast<const infinicore::Tensor &>(weight_scale_);
        std::optional<infinicore::Tensor> bias_opt = has_bias_ ? std::make_optional<infinicore::Tensor>(static_cast<const infinicore::Tensor &>(bias_)) : std::nullopt;

        auto output = infinicore::op::linear_w8a8i8(input_contiguous->contiguous(), weight_packed_tensor, weight_scale_tensor, bias_opt);
        return output;
    }
    case infinilm::quantization::QuantScheme::AWQ_W4A16: {
        infinicore::Tensor input_contiguous = input->is_contiguous() ? input : input->contiguous();
        infinicore::Tensor qweight = static_cast<const infinicore::Tensor &>(weight_);
        infinicore::Tensor qzeros = static_cast<const infinicore::Tensor &>(weight_zeros_);
        infinicore::Tensor scales = static_cast<const infinicore::Tensor &>(weight_scale_);
        std::optional<infinicore::Tensor> bias_opt = has_bias_ ? std::make_optional<infinicore::Tensor>(static_cast<const infinicore::Tensor &>(bias_)) : std::nullopt;
        auto output = infinicore::op::linear_w4a16_awq(input_contiguous->contiguous(), qweight, scales, qzeros, bias_opt);
        return output;
    }
    case infinilm::quantization::QuantScheme::GPTQ_W4A16_QY: {
        infinicore::Tensor input_contiguous = input->is_contiguous() ? input : input->contiguous();
        infinicore::Tensor qweight = static_cast<const infinicore::Tensor &>(weight_);
        infinicore::Tensor qzeros = static_cast<const infinicore::Tensor &>(weight_zeros_);
        infinicore::Tensor scales = static_cast<const infinicore::Tensor &>(weight_scale_);
        infinicore::Tensor g_idx = static_cast<const infinicore::Tensor &>(gidx_);
        std::optional<infinicore::Tensor> bias_opt = has_bias_ ? std::make_optional<infinicore::Tensor>(static_cast<const infinicore::Tensor &>(bias_)) : std::nullopt;
        auto output = infinicore::op::linear_w4a16_gptq_qy(input_contiguous->contiguous(), qweight, qzeros, scales, 0, 4);
        if (bias_opt.has_value()) {
            infinicore::op::add_(output, output, bias_opt.value()->as_strided(output->shape(), {0, 0, 1}));
        }
        return output;
    }
    case infinilm::quantization::QuantScheme::GPTQ_W4A16: {
        throw std::runtime_error("GPTQ_W4A16 quantization scheme is not yet supported in forward pass");
    }
    default: {
        infinicore::Tensor input_contiguous = input->is_contiguous() ? input : input->contiguous();

        infinicore::Tensor weight_tensor = static_cast<const infinicore::Tensor &>(weight_);
        std::optional<infinicore::Tensor> bias_opt = has_bias_ ? std::make_optional<infinicore::Tensor>(static_cast<const infinicore::Tensor &>(bias_)) : std::nullopt;

        auto output = infinicore::op::linear(input_contiguous->contiguous(), weight_tensor->contiguous(), bias_opt);
        return output;
    }
    }

} // namespace infinilm::nn

infinicore::Tensor BaseLinear::forward(infinicore::Tensor &input) const {
    return compute_linear(input);
}

infinicore::Tensor BaseLinear::forward(infinicore::Tensor &input, infinicore::Tensor &residual) const {
    auto output = compute_linear(input);

    infinicore::op::add_(output, output, residual);

    return output;
}

void BaseLinear::process_weights_after_loading() {
    if (quantization_->get_quant_scheme() == infinilm::quantization::QuantScheme::GPTQ_W4A16) {

        auto config = quantization_->get_config();
        auto gptq_qy = std::make_shared<infinilm::quantization::GPTQ_QY>(config);
        quantization_ = gptq_qy;

        {
            auto orig_weight = weight_;
            auto orig_zeros = weight_zeros_;
            auto orig_scales = weight_scale_;

            gptq_qy->convert_from_gptq_w4a16(
                orig_weight, orig_zeros, orig_scales, gidx_, device_);
        }

        weight_.reset();
        weight_zeros_.reset();
        weight_scale_.reset();

        weight_ = gptq_qy->get_converted_weight();
        weight_zeros_ = gptq_qy->get_converted_zeros();
        weight_scale_ = gptq_qy->get_converted_scales();
        gptq_qy->release_buffers();

        assert(quantization_->get_quant_scheme() == infinilm::quantization::QuantScheme::GPTQ_W4A16_QY);
    }
}
} // namespace infinilm::nn
