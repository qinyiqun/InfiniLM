#pragma once

#include "base_quantization.hpp"
namespace infinilm::quantization {

class NoneQuantization : public BaseQuantization {
public:
    explicit NoneQuantization(const nlohmann::json &quant_config)
        : BaseQuantization(quant_config) {};

    infinilm::quantization::QuantScheme
    get_quant_scheme() const override {
        return infinilm::quantization::QuantScheme::NONE;
    };
};

} // namespace infinilm::quantization
