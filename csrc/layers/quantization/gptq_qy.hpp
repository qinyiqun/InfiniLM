#pragma once

#include "infinicore/tensor.hpp"
#include "base_quantization.hpp"
#include <algorithm>
#include <cassert>
#include <cstring>
#include <memory>
#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>
#include <vector>

namespace {
#ifndef INFINICORE_FLOAT16_DEFINED
#define INFINICORE_FLOAT16_DEFINED
struct float16_raw {
    uint16_t data;
    float16_raw() : data(0) {}
    explicit float16_raw(float f) : data(fp32_to_fp16_bits(f)) {}

    static uint16_t fp32_to_fp16_bits(float value) {
        union {
            float f;
            uint32_t u;
        } f2u;
        f2u.f = value;
        uint32_t x = f2u.u;

        uint32_t sign = (x >> 16) & 0x8000;
        int32_t exp = ((x >> 23) & 0xFF) - 127;
        uint32_t mantissa = x & 0x007FFFFF;

        if (exp == 128) {
            if (mantissa == 0) {
                return static_cast<uint16_t>(sign | 0x7C00);
            } else {
                return static_cast<uint16_t>(sign | 0x7C00 | (mantissa >> 13));
            }
        }
        if (exp > 15) {
            return static_cast<uint16_t>(sign | 0x7C00);
        }
        if (exp < -14) {
            if (exp < -24) {
                return static_cast<uint16_t>(sign);
            }
            mantissa |= 0x00800000;
            uint32_t shift = -exp - 14;
            mantissa >>= shift;
            if ((mantissa & 0x1000) && ((mantissa & 0x2FFF) != 0)) {
                mantissa += 0x2000;
            }
            return static_cast<uint16_t>(sign | (mantissa >> 13));
        }

        uint32_t exp16 = static_cast<uint32_t>(exp + 15) << 10;
        uint32_t mantissa16 = mantissa >> 13;
        if ((mantissa & 0x1000) && ((mantissa & 0x2FFF) || (mantissa16 & 1))) {
            mantissa16++;
            if (mantissa16 == 0x400) {
                exp16 += 0x400;
                mantissa16 = 0;
            }
        }
        return static_cast<uint16_t>(sign | exp16 | mantissa16);
    }
};
using float16_t = float16_raw;
#endif

inline std::vector<uint16_t> float_to_fp16_bits(const std::vector<float> &values) {
    std::vector<uint16_t> result;
    result.reserve(values.size());
    for (float f : values) {
#ifdef INFINICORE_HAS_FLOAT16
        infinicore::float16_t h(f);
        result.push_back(*reinterpret_cast<uint16_t *>(&h));
#else
        result.push_back(float16_raw::fp32_to_fp16_bits(f));
#endif
    }
    return result;
}
} // anonymous namespace

namespace infinilm::quantization {

class GPTQ_QY : public BaseQuantization {
public:
    explicit GPTQ_QY(const nlohmann::json &quant_config)
        : BaseQuantization(quant_config) {
        int bits = weight_bits();
        if (bits != 4) {
            spdlog::warn("GPTQ_QY: bits={} not fully tested, expected 4", bits);
        }
    }

    QuantScheme get_quant_scheme() const override {
        return QuantScheme::GPTQ_W4A16_QY;
    }

    int get_packing_num() const {
        return 32 / weight_bits();
    }

    int get_group_size() const {
        return get_or<int>("group_size", 128);
    }

    int weight_bits() const { return get_or<int>("bits", 4); }
    bool desc_act() const { return get_or<bool>("desc_act", false); }

    // Parameter layout for GPTQ_QY (already converted format)
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

    // Split fused linear parameters into named sub-parameters
    std::vector<SplitParam> split_params(
        const std::unordered_map<std::string, infinicore::nn::Parameter> &params,
        const std::vector<SplitInfo> &splits,
        int narrow_dim,
        int tp_rank, int tp_size, int tp_num_heads) const override;

    // Convert from GPTQ_W4A16 format to GPTQ_QY format
    void convert_from_gptq_w4a16(const infinicore::Tensor &original_qweight,
                                 const infinicore::Tensor &original_qzeros,
                                 const infinicore::Tensor &original_scales,
                                 const infinicore::Tensor &g_idx,
                                 const infinicore::Device &target_device);

    void release_buffers() {
        converted_weight_ = infinicore::Tensor();
        converted_zeros_ = infinicore::Tensor();
        converted_scales_ = infinicore::Tensor();
        g_idx_ = infinicore::Tensor();
    }

    const infinicore::Tensor &get_converted_weight() const { return std::move(converted_weight_); }
    const infinicore::Tensor &get_converted_zeros() const { return std::move(converted_zeros_); }
    const infinicore::Tensor &get_converted_scales() const { return std::move(converted_scales_); }
    const infinicore::Tensor &get_g_idx() const { return g_idx_; }
    bool is_converted() const { return converted_; }

private:
    static std::vector<uint8_t> unpack_int32_to_nibbles_3d_(const infinicore::Tensor &packed, int bits);
    static std::vector<uint8_t> combine_nibbles_last_dim_(const std::vector<uint8_t> &nibbles, size_t M, size_t K, size_t N);
    static std::vector<float> unpack_zeros_to_fp32_2d_(const infinicore::Tensor &packed_zeros, int bits);
    static infinicore::Tensor make_tensor_from_host_(const void *data, size_t bytes,
                                                     const std::vector<size_t> &shape,
                                                     infinicore::DataType dtype, const infinicore::Device &device);

    infinicore::Tensor converted_weight_;
    infinicore::Tensor converted_zeros_;
    infinicore::Tensor converted_scales_;
    infinicore::Tensor g_idx_;
    bool converted_ = false;
};

} // namespace infinilm::quantization
