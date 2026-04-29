#pragma once
#include "base_quantization.hpp"
namespace infinilm::quantization {

class AWQ : public BaseQuantization {
public:
    explicit AWQ(const nlohmann::json &quant_config)
        : BaseQuantization(quant_config){};

    infinilm::quantization::QuantScheme
    get_quant_scheme() const override {
        return infinilm::quantization::QuantScheme::AWQ_W4A16;
    };

    int get_packing_num() const {
        return 32 / this->get_or<int>("bits", 4);
    }

    int get_group_size() const {
        return this->get_or<int>("group_size", 128);
    }
};

} // namespace infinilm::quantization
