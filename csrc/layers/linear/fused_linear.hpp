#pragma once
#include "../../engine/distributed/communication_group.hpp"
#include "linear.hpp"
#include "../quantization/quantization.hpp"
#include <functional>

namespace infinilm::layers::linear {
class QKVParallelLinear : public infinilm::nn::ColumnParallelLinear {
public:
    explicit QKVParallelLinear(size_t hidden_size,
                               size_t q_dim, size_t k_dim, size_t v_dim,
                               size_t num_q_head, size_t num_k_head, size_t num_v_head,
                               bool q_bias, bool k_bias, bool v_bias,
                               const infinicore::DataType &dtype = infinicore::DataType::F32,
                               const infinicore::Device &device = infinicore::Device(),
                               engine::distributed::RankInfo rank_info = engine::distributed::RankInfo());

    // A more common case where all heads have the same dimension
    explicit QKVParallelLinear(size_t hidden_size,
                               size_t head_dim,
                               size_t num_q_head, size_t num_kv_head,
                               bool bias = false,
                               const infinicore::DataType &dtype = infinicore::DataType::F32,
                               const infinicore::Device &device = infinicore::Device(),
                               engine::distributed::RankInfo rank_info = engine::distributed::RankInfo());

    explicit QKVParallelLinear(size_t hidden_size,
                               size_t q_dim, size_t k_dim, size_t v_dim,
                               size_t num_q_head, size_t num_k_head, size_t num_v_head,
                               bool q_bias, bool k_bias, bool v_bias,
                               std::shared_ptr<infinilm::quantization::BaseQuantization> quantization,
                               const infinicore::DataType &dtype = infinicore::DataType::F32,
                               const infinicore::Device &device = infinicore::Device(),
                               engine::distributed::RankInfo rank_info = engine::distributed::RankInfo());

    // A more common case where all heads have the same dimension
    explicit QKVParallelLinear(size_t hidden_size,
                               size_t head_dim,
                               size_t num_q_head, size_t num_kv_head,
                               std::shared_ptr<infinilm::quantization::BaseQuantization> quantization,
                               bool bias = false,
                               const infinicore::DataType &dtype = infinicore::DataType::F32,
                               const infinicore::Device &device = infinicore::Device(),
                               engine::distributed::RankInfo rank_info = engine::distributed::RankInfo());

    std::tuple<infinicore::Tensor, infinicore::Tensor, infinicore::Tensor>
    forward_split(infinicore::Tensor &input);

    infinicore::nn::Parameter get_q_weight() const;
    infinicore::nn::Parameter get_k_weight() const;
    infinicore::nn::Parameter get_v_weight() const;

    infinicore::nn::Parameter get_q_weight_scale() const;
    infinicore::nn::Parameter get_k_weight_scale() const;
    infinicore::nn::Parameter get_v_weight_scale() const;

    infinicore::nn::Parameter get_q_weight_zeros() const;
    infinicore::nn::Parameter get_k_weight_zeros() const;
    infinicore::nn::Parameter get_v_weight_zeros() const;

    // For computing the packing factor in awq quantization:
    // Returns the number of low-bit elements packed into a single high-bit container element.
    // For example: int4 → int32 yields a packing factor of 8 (32 bits / 4 bits = 8 int4 values per int32).
    infinicore::nn::Parameter get_q_weight_awq(int scaling_factor) const;
    infinicore::nn::Parameter get_k_weight_awq(int scaling_factor) const;
    infinicore::nn::Parameter get_v_weight_awq(int scaling_factor) const;

    infinicore::nn::Parameter get_q_weight_scale_awq(int scaling_factor) const;
    infinicore::nn::Parameter get_k_weight_scale_awq(int scaling_factor) const;
    infinicore::nn::Parameter get_v_weight_scale_awq(int scaling_factor) const;

    infinicore::nn::Parameter get_q_weight_zeros_awq(int scaling_factor) const;
    infinicore::nn::Parameter get_k_weight_zeros_awq(int scaling_factor) const;
    infinicore::nn::Parameter get_v_weight_zeros_awq(int scaling_factor) const;

    infinicore::nn::Parameter get_q_bias() const;
    infinicore::nn::Parameter get_k_bias() const;
    infinicore::nn::Parameter get_v_bias() const;

    infinicore::nn::Parameter get_q_g_idx_gptq() const;
    infinicore::nn::Parameter get_k_g_idx_gptq() const;
    infinicore::nn::Parameter get_v_g_idx_gptq() const;

    bool has_q_bias() const;
    bool has_k_bias() const;
    bool has_v_bias() const;

    void register_parameters(std::function<void(const std::string &, infinicore::nn::Parameter)> register_fn,
                             const std::string &q_name,
                             const std::string &k_name,
                             const std::string &v_name);

private:
    static size_t calculate_kv_replicas(size_t num_k_head, size_t tp_size) {
        if (num_k_head % tp_size == 0) {
            return 1;
        }
        if (tp_size % num_k_head == 0) {
            return (tp_size + num_k_head - 1) / num_k_head;
        }
        throw std::runtime_error("Invalid KV head configuration");
    }

    static size_t
    calculate_out_feature_size(size_t num_q_head, size_t q_dim, size_t num_k_head, size_t k_dim, size_t num_v_head, size_t v_dim, engine::distributed::RankInfo rank_info) {
        return num_q_head * q_dim + num_k_head * k_dim * calculate_kv_replicas(num_k_head, rank_info.tp_size) + num_v_head * v_dim * calculate_kv_replicas(num_v_head, rank_info.tp_size);
    }

private:
    size_t q_dim_;
    size_t k_dim_;
    size_t v_dim_;
    size_t num_q_head_;
    size_t num_k_head_;
    size_t num_v_head_;
    bool q_bias_;
    bool k_bias_;
    bool v_bias_;
    size_t q_out_size_; // num_q_head * q_dim / tp_size
    size_t k_out_size_; // num_k_head * k_dim / tp_size
    size_t v_out_size_; // num_v_head * v_dim / tp_size

    size_t num_kv_head_replicas_ = 1;
};

class GateUpParallelLinear : public infinilm::nn::ColumnParallelLinear {
public:
    /**
     * @deprecated This function is deprecated and will be REMOVED in the next major release (v0.2.0).
     *
     * ⚠️ DEVELOPMENT POLICY:
     *   - NO new development or feature additions permitted on this interface
     *   - Only critical bug fixes (security/stability) allowed until removal
     *   - All new code MUST migrate to the polymorphic overload below
     *
     * Replacement: Use the polymorphic overload of this same function name with updated signature
     * Reason: Legacy signature lacks support for dynamic quantization modes.
     * Removal target: v0.2.0 (Q2 2026)
     */
    GateUpParallelLinear(size_t hidden_size, size_t intermediate_size, bool bias = false,
                         const infinicore::DataType &dtype = infinicore::DataType::F32, const infinicore::Device &device = infinicore::Device(),
                         engine::distributed::RankInfo rank_info = engine::distributed::RankInfo());

    GateUpParallelLinear(size_t hidden_size, size_t intermediate_size, bool gate_bias, bool up_bias,
                         const infinicore::DataType &dtype = infinicore::DataType::F32, const infinicore::Device &device = infinicore::Device(),
                         engine::distributed::RankInfo rank_info = engine::distributed::RankInfo());

    GateUpParallelLinear(size_t hidden_size, size_t intermediate_size, std::shared_ptr<infinilm::quantization::BaseQuantization> quantization,
                         bool bias = false,
                         const infinicore::DataType &dtype = infinicore::DataType::F32,
                         const infinicore::Device &device = infinicore::Device(),
                         engine::distributed::RankInfo rank_info = engine::distributed::RankInfo());

    GateUpParallelLinear(size_t hidden_size, size_t intermediate_size, bool gate_bias, bool up_bias,
                         std::shared_ptr<infinilm::quantization::BaseQuantization> quantization,
                         const infinicore::DataType &dtype = infinicore::DataType::F32, const infinicore::Device &device = infinicore::Device(),
                         engine::distributed::RankInfo rank_info = engine::distributed::RankInfo());

    std::tuple<infinicore::Tensor, infinicore::Tensor> forward_split(infinicore::Tensor &input);

    infinicore::nn::Parameter get_gate_weight() const;

    infinicore::nn::Parameter get_gate_weight_scale() const;

    infinicore::nn::Parameter get_gate_weight_zeros() const;

    infinicore::nn::Parameter get_gate_bias() const;

    infinicore::nn::Parameter get_up_weight() const;

    infinicore::nn::Parameter get_up_weight_scale() const;

    infinicore::nn::Parameter get_up_weight_zeros() const;

    infinicore::nn::Parameter get_up_bias() const;

    infinicore::nn::Parameter get_gate_weight_awq() const;

    infinicore::nn::Parameter get_up_weight_awq() const;

    infinicore::nn::Parameter get_up_weight_scale_awq() const;

    infinicore::nn::Parameter get_up_weight_zeros_awq() const;

    infinicore::nn::Parameter get_gate_weight_scale_awq() const;

    infinicore::nn::Parameter get_gate_weight_zeros_awq() const;

    infinicore::nn::Parameter get_gate_g_idx_gptq() const;

    infinicore::nn::Parameter get_up_g_idx_gptq() const;

    bool has_gate_bias() const;

    bool has_up_bias() const;

    void register_parameters(std::function<void(const std::string &, infinicore::nn::Parameter)> register_fn,
                             const std::string &gate_name,
                             const std::string &up_name);

private:
    bool gate_bias_;
    bool up_bias_;
};

} // namespace infinilm::layers::linear
