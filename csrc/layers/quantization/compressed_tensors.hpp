#pragma once

#include "base_quantization.hpp"
namespace infinilm::quantization {

class CompressedTensors : public BaseQuantization {
public:
    explicit CompressedTensors(const nlohmann::json &quant_config)
        : BaseQuantization(quant_config) {};

    infinilm::quantization::QuantScheme
    get_quant_scheme() const override {
        return infinilm::quantization::QuantScheme::COMPRESSED_TENSOR_W8A8I8;
    };
};

} // namespace infinilm::quantization
