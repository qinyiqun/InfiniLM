#pragma once
#include "nlohmann/json.hpp"
#include "quantization_scheme.hpp"

namespace infinilm::quantization {
class BaseQuantization {
public:
    explicit BaseQuantization(const nlohmann::json &quant_config) : quant_config_(quant_config) {};
    virtual ~BaseQuantization() = default;

    const nlohmann::json &get_config() const { return quant_config_; }

    virtual infinilm::quantization::QuantScheme get_quant_scheme() const = 0;
    template <typename T>
    T get(const std::string &key) const {
        if (!quant_config_.contains(key)) {
            throw std::out_of_range("Key '" + key + "' not found in config.");
        }
        try {
            return quant_config_.at(key).get<T>();
        } catch (const nlohmann::json::type_error &e) {
            throw std::runtime_error("Type conversion failed for key '" + key + "': " + std::string(e.what()));
        }
    }

    template <typename T>
    T get_or(const std::string &key, const T &default_value) const {
        if (!quant_config_.contains(key) || quant_config_.at(key).is_null()) {
            return default_value;
        }
        try {
            return quant_config_.at(key).get<T>();
        } catch (const nlohmann::json::type_error &) {
            return default_value;
        }
    }

protected:
    nlohmann::json quant_config_;
};
} // namespace infinilm::quantization
