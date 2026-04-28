#include "fused_linear.hpp"

#include <spdlog/spdlog.h>

namespace infinilm::layers::linear {
// ---------------------------------------------------------
// QKV Parallel Linear
// ---------------------------------------------------------
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
QKVParallelLinear::QKVParallelLinear(size_t hidden_size,
                                     size_t head_dim,
                                     size_t num_q_head,
                                     size_t num_kv_head,
                                     bool bias,
                                     const infinicore::DataType &dtype,
                                     const infinicore::Device &device,
                                     engine::distributed::RankInfo rank_info)
    : QKVParallelLinear(hidden_size,
                        head_dim, head_dim, head_dim,
                        num_q_head, num_kv_head, num_kv_head,
                        bias, bias, bias,
                        dtype, device, rank_info) {}

QKVParallelLinear::QKVParallelLinear(size_t hidden_size,
                                     size_t q_dim, size_t k_dim, size_t v_dim,
                                     size_t num_q_head, size_t num_k_head, size_t num_v_head,
                                     bool q_bias, bool k_bias, bool v_bias,
                                     const infinicore::DataType &dtype,
                                     const infinicore::Device &device,
                                     engine::distributed::RankInfo rank_info)
    : infinicore::nn::ColumnParallelLinear(
          hidden_size,
          num_q_head * q_dim + num_k_head * k_dim + num_v_head * v_dim,
          (q_bias || k_bias || v_bias),
          dtype,
          device,
          rank_info.tp_rank,
          rank_info.tp_size),
      q_dim_(q_dim),
      k_dim_(k_dim),
      v_dim_(v_dim),
      num_q_head_(num_q_head),
      num_k_head_(num_k_head),
      num_v_head_(num_v_head),
      q_bias_(q_bias),
      k_bias_(k_bias),
      v_bias_(v_bias) {
    if (num_q_head % tp_size_ != 0 || num_k_head % tp_size_ != 0 || num_v_head % tp_size_ != 0) {
        throw std::runtime_error("QKVParallelLinear: num_[q|k|v]_head must be divisible by tp_size");
    }

    if ((q_bias_ != k_bias_) || (k_bias_ != v_bias_)) {
        throw std::runtime_error("q_bias, k_bias, v_bias must all match");
    }

    q_out_size_ = num_q_head_ * q_dim_ / tp_size_;
    k_out_size_ = num_k_head_ * k_dim_ / tp_size_;
    v_out_size_ = num_v_head_ * v_dim_ / tp_size_;
}

QKVParallelLinear::QKVParallelLinear(size_t hidden_size,
                                     size_t head_dim,
                                     size_t num_q_head,
                                     size_t num_kv_head,
                                     std::shared_ptr<infinicore::quantization::BaseQuantization> quantization,
                                     bool bias,
                                     const infinicore::DataType &dtype,
                                     const infinicore::Device &device,
                                     engine::distributed::RankInfo rank_info)
    : QKVParallelLinear(hidden_size,
                        head_dim, head_dim, head_dim,
                        num_q_head, num_kv_head, num_kv_head,
                        bias, bias, bias,
                        quantization,
                        dtype, device, rank_info) {}

QKVParallelLinear::QKVParallelLinear(size_t hidden_size,
                                     size_t q_dim, size_t k_dim, size_t v_dim,
                                     size_t num_q_head, size_t num_k_head, size_t num_v_head,
                                     bool q_bias, bool k_bias, bool v_bias,
                                     std::shared_ptr<infinicore::quantization::BaseQuantization> quantization,
                                     const infinicore::DataType &dtype,
                                     const infinicore::Device &device,
                                     engine::distributed::RankInfo rank_info)
    : infinicore::nn::ColumnParallelLinear(
          hidden_size,
          calculate_out_feature_size(num_q_head, q_dim, num_k_head, k_dim, num_v_head, v_dim, rank_info),
          quantization,
          (q_bias || k_bias || v_bias),
          dtype,
          device,
          rank_info.tp_rank,
          rank_info.tp_size),
      q_dim_(q_dim),
      k_dim_(k_dim),
      v_dim_(v_dim),
      num_q_head_(num_q_head),
      num_k_head_(num_k_head),
      num_v_head_(num_v_head),
      q_bias_(q_bias),
      k_bias_(k_bias),
      v_bias_(v_bias),
      num_kv_head_replicas_(calculate_kv_replicas(num_k_head, rank_info.tp_size)) {

    if ((q_bias_ != k_bias_) || (k_bias_ != v_bias_)) {
        throw std::runtime_error("q_bias, k_bias, v_bias must all match");
    }

    q_out_size_ = num_q_head_ * q_dim_ / tp_size_;
    k_out_size_ = num_kv_head_replicas_ * num_k_head_ * k_dim_ / tp_size_;
    v_out_size_ = num_kv_head_replicas_ * num_v_head_ * v_dim_ / tp_size_;
}

std::tuple<infinicore::Tensor, infinicore::Tensor, infinicore::Tensor>
QKVParallelLinear::forward_split(infinicore::Tensor &input) {
    auto output = this->forward(input);

    auto q_out = output->narrow({{2, 0, q_out_size_}});
    auto k_out = output->narrow({{2, q_out_size_, k_out_size_}});
    auto v_out = output->narrow({{2, q_out_size_ + k_out_size_, v_out_size_}});

    return std::make_tuple(q_out, k_out, v_out);
}

infinicore::nn::Parameter QKVParallelLinear::get_q_weight() const {
    return infinicore::nn::Parameter(
        weight_->narrow({{0, 0, q_out_size_}}),
        0, tp_rank_, tp_size_);
}

infinicore::nn::Parameter QKVParallelLinear::get_k_weight() const {
    return infinicore::nn::Parameter(
        weight_->narrow({{0, q_out_size_, k_out_size_}}),
        0, tp_rank_, tp_size_, num_k_head_);
}

infinicore::nn::Parameter QKVParallelLinear::get_v_weight() const {
    return infinicore::nn::Parameter(
        weight_->narrow({{0, q_out_size_ + k_out_size_, v_out_size_}}),
        0, tp_rank_, tp_size_, num_v_head_);
}

infinicore::nn::Parameter QKVParallelLinear::get_q_weight_scale() const {
    return infinicore::nn::Parameter(
        weight_scale_->narrow({{0, 0, q_out_size_}}), 0, tp_rank_, tp_size_);
}

infinicore::nn::Parameter QKVParallelLinear::get_k_weight_scale() const {
    return infinicore::nn::Parameter(
        weight_scale_->narrow({{0, q_out_size_, k_out_size_}}),
        0, tp_rank_, tp_size_, num_k_head_);
}

infinicore::nn::Parameter QKVParallelLinear::get_v_weight_scale() const {
    return infinicore::nn::Parameter(
        weight_scale_->narrow({{0, q_out_size_ + k_out_size_, v_out_size_}}),
        0, tp_rank_, tp_size_, num_k_head_);
}

infinicore::nn::Parameter QKVParallelLinear::get_q_weight_awq(int scaling_factor) const {
    return infinicore::nn::Parameter(
        weight_->narrow({{1, 0, q_out_size_ / scaling_factor}}),
        1, tp_rank_, tp_size_);
}

infinicore::nn::Parameter QKVParallelLinear::get_k_weight_awq(int scaling_factor) const {
    return infinicore::nn::Parameter(
        weight_->narrow({{1, q_out_size_ / scaling_factor, k_out_size_ / scaling_factor}}),
        1, tp_rank_, tp_size_, num_k_head_);
}

infinicore::nn::Parameter QKVParallelLinear::get_v_weight_awq(int scaling_factor) const {
    return infinicore::nn::Parameter(
        weight_->narrow({{1, (q_out_size_ + k_out_size_) / scaling_factor, v_out_size_ / scaling_factor}}),
        1, tp_rank_, tp_size_, num_k_head_);
}

infinicore::nn::Parameter QKVParallelLinear::get_q_weight_scale_awq(int scaling_factor) const {
    return infinicore::nn::Parameter(
        weight_scale_->narrow({{1, 0, q_out_size_ / scaling_factor}}), 1, tp_rank_, tp_size_);
}

infinicore::nn::Parameter QKVParallelLinear::get_k_weight_scale_awq(int scaling_factor) const {
    return infinicore::nn::Parameter(
        weight_scale_->narrow({{1, q_out_size_ / scaling_factor, k_out_size_ / scaling_factor}}),
        1, tp_rank_, tp_size_, num_k_head_);
}

infinicore::nn::Parameter QKVParallelLinear::get_v_weight_scale_awq(int scaling_factor) const {
    return infinicore::nn::Parameter(
        weight_scale_->narrow({{1, (q_out_size_ + k_out_size_) / scaling_factor, v_out_size_ / scaling_factor}}),
        1, tp_rank_, tp_size_, num_k_head_);
}

infinicore::nn::Parameter QKVParallelLinear::get_q_weight_zeros_awq(int scaling_factor) const {
    return infinicore::nn::Parameter(
        weight_zeros_->narrow({{1, 0, q_out_size_ / scaling_factor}}), 1, tp_rank_, tp_size_);
}

infinicore::nn::Parameter QKVParallelLinear::get_k_weight_zeros_awq(int scaling_factor) const {
    return infinicore::nn::Parameter(
        weight_zeros_->narrow({{1, q_out_size_ / scaling_factor, k_out_size_ / scaling_factor}}),
        1, tp_rank_, tp_size_, num_k_head_);
}

infinicore::nn::Parameter QKVParallelLinear::get_v_weight_zeros_awq(int scaling_factor) const {
    return infinicore::nn::Parameter(
        weight_zeros_->narrow({{1, (q_out_size_ + k_out_size_) / scaling_factor, v_out_size_ / scaling_factor}}),
        1, tp_rank_, tp_size_, num_k_head_);
}

infinicore::nn::Parameter QKVParallelLinear::get_q_weight_zeros() const {
    return infinicore::nn::Parameter(
        weight_zeros_->narrow({{0, 0, q_out_size_}}), 0, tp_rank_, tp_size_);
}

infinicore::nn::Parameter QKVParallelLinear::get_k_weight_zeros() const {
    return infinicore::nn::Parameter(
        weight_zeros_->narrow({{0, q_out_size_, k_out_size_}}),
        0, tp_rank_, tp_size_, num_k_head_);
}

infinicore::nn::Parameter QKVParallelLinear::get_v_weight_zeros() const {
    return infinicore::nn::Parameter(
        weight_zeros_->narrow({{0, q_out_size_ + k_out_size_, v_out_size_}}),
        0, tp_rank_, tp_size_, num_k_head_);
}

infinicore::nn::Parameter QKVParallelLinear::get_q_bias() const {
    if (!q_bias_) {
        return infinicore::nn::Parameter();
    }
    return infinicore::nn::Parameter(
        bias_->narrow({{0, 0, q_out_size_}}),
        0, tp_rank_, tp_size_);
}

infinicore::nn::Parameter QKVParallelLinear::get_k_bias() const {
    if (!k_bias_) {
        return infinicore::nn::Parameter();
    }
    return infinicore::nn::Parameter(
        bias_->narrow({{0, q_out_size_, k_out_size_}}),
        0, tp_rank_, tp_size_);
}

infinicore::nn::Parameter QKVParallelLinear::get_v_bias() const {
    if (!v_bias_) {
        return infinicore::nn::Parameter();
    }
    return infinicore::nn::Parameter(
        bias_->narrow({{0, q_out_size_ + k_out_size_, v_out_size_}}),
        0, tp_rank_, tp_size_);
}

infinicore::nn::Parameter QKVParallelLinear::get_q_g_idx_gptq() const {
    return infinicore::nn::Parameter(gidx_->narrow({{0, 0, in_features_ / tp_size_}}), 0, tp_rank_, tp_size_);
}

infinicore::nn::Parameter QKVParallelLinear::get_k_g_idx_gptq() const {
    return infinicore::nn::Parameter(gidx_->narrow({{0, 0, in_features_ / tp_size_}}), 0, tp_rank_, tp_size_, num_k_head_);
}

infinicore::nn::Parameter QKVParallelLinear::get_v_g_idx_gptq() const {
    return infinicore::nn::Parameter(gidx_->narrow({{0, 0, in_features_ / tp_size_}}), 0, tp_rank_, tp_size_, num_k_head_);
}

bool QKVParallelLinear::has_q_bias() const { return q_bias_; }
bool QKVParallelLinear::has_k_bias() const { return k_bias_; }
bool QKVParallelLinear::has_v_bias() const { return v_bias_; }

// ---------------------------------------------------------
// Gate-Up Parallel Linear
// ---------------------------------------------------------
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
GateUpParallelLinear::GateUpParallelLinear(size_t hidden_size, size_t intermediate_size, bool bias,
                                           const infinicore::DataType &dtype, const infinicore::Device &device,
                                           engine::distributed::RankInfo rank_info)
    : GateUpParallelLinear(hidden_size, intermediate_size, bias, bias, dtype, device, rank_info) {
}

GateUpParallelLinear::GateUpParallelLinear(size_t hidden_size, size_t intermediate_size, bool gate_bias, bool up_bias,
                                           const infinicore::DataType &dtype, const infinicore::Device &device,
                                           engine::distributed::RankInfo rank_info)
    : infinicore::nn::ColumnParallelLinear(hidden_size, intermediate_size * 2, gate_bias || up_bias, dtype, device, rank_info.tp_rank, rank_info.tp_size), gate_bias_(gate_bias), up_bias_(up_bias) {
    if (gate_bias_ != up_bias_) {
        throw std::runtime_error("Not supported yet: gate_bias and up_bias should be given at the same time");
    }
}

GateUpParallelLinear::GateUpParallelLinear(size_t hidden_size, size_t intermediate_size, std::shared_ptr<infinicore::quantization::BaseQuantization> quantization, bool bias,
                                           const infinicore::DataType &dtype, const infinicore::Device &device,
                                           engine::distributed::RankInfo rank_info)
    : GateUpParallelLinear(hidden_size, intermediate_size, bias, bias, quantization, dtype, device, rank_info) {
}

GateUpParallelLinear::GateUpParallelLinear(size_t hidden_size, size_t intermediate_size, bool gate_bias, bool up_bias,
                                           std::shared_ptr<infinicore::quantization::BaseQuantization> quantization,
                                           const infinicore::DataType &dtype, const infinicore::Device &device,
                                           engine::distributed::RankInfo rank_info)
    : infinicore::nn::ColumnParallelLinear(hidden_size, intermediate_size * 2, quantization, gate_bias || up_bias, dtype, device, rank_info.tp_rank, rank_info.tp_size), gate_bias_(gate_bias), up_bias_(up_bias) {
    if (gate_bias_ != up_bias_) {
        throw std::runtime_error("Not supported yet: gate_bias and up_bias should be given at the same time");
    }
}

std::tuple<infinicore::Tensor, infinicore::Tensor> GateUpParallelLinear::forward_split(infinicore::Tensor &input) {
    auto output = this->forward(input);
    auto cols = output->shape()[2];
    auto gate_output = output->narrow({{2, 0, cols / 2}});
    auto up_output = output->narrow({{2, cols / 2, cols / 2}});
    return std::make_tuple(gate_output, up_output);
}

infinicore::nn::Parameter GateUpParallelLinear::get_gate_weight() const {
    return infinicore::nn::Parameter(weight_->narrow({{0, 0, weight_->size(0) / 2}}), 0, tp_rank_, tp_size_);
}

infinicore::nn::Parameter GateUpParallelLinear::get_gate_bias() const {
    if (!gate_bias_) {
        return infinicore::nn::Parameter();
    } else {
        return infinicore::nn::Parameter(bias_->narrow({{0, 0, bias_->size(0) / 2}}), 0, tp_rank_, tp_size_);
    }
}

infinicore::nn::Parameter GateUpParallelLinear::get_up_weight() const {
    return infinicore::nn::Parameter(weight_->narrow({{0, weight_->size(0) / 2, weight_->size(0) / 2}}), 0, tp_rank_, tp_size_);
}

infinicore::nn::Parameter GateUpParallelLinear::get_up_bias() const {
    if (!up_bias_) {
        return infinicore::nn::Parameter();
    } else {
        return infinicore::nn::Parameter(bias_->narrow({{0, bias_->size(0) / 2, bias_->size(0) / 2}}),
                                         0, tp_rank_, tp_size_);
    }
}

infinicore::nn::Parameter GateUpParallelLinear::get_gate_weight_scale() const {
    return infinicore::nn::Parameter(weight_scale_->narrow({{0, 0, weight_scale_->size(0) / 2}}), 0, tp_rank_, tp_size_);
}

infinicore::nn::Parameter GateUpParallelLinear::get_up_weight_scale() const {
    return infinicore::nn::Parameter(weight_scale_->narrow({{0, weight_scale_->size(0) / 2, weight_scale_->size(0) / 2}}), 0, tp_rank_, tp_size_);
}

infinicore::nn::Parameter GateUpParallelLinear::get_gate_weight_zeros() const {
    return infinicore::nn::Parameter(weight_zeros_->narrow({{0, 0, weight_zeros_->size(0) / 2}}), 0, tp_rank_, tp_size_);
}

infinicore::nn::Parameter GateUpParallelLinear::get_up_weight_zeros() const {
    return infinicore::nn::Parameter(weight_zeros_->narrow({{0, weight_zeros_->size(0) / 2, weight_zeros_->size(0) / 2}}), 0, tp_rank_, tp_size_);
}

bool GateUpParallelLinear::has_gate_bias() const {
    return gate_bias_;
}

bool GateUpParallelLinear::has_up_bias() const {
    return up_bias_;
}

infinicore::nn::Parameter GateUpParallelLinear::get_gate_weight_awq() const {
    return infinicore::nn::Parameter(weight_->narrow({{1, 0, weight_->size(1) / 2}}), 1, tp_rank_, tp_size_);
}

infinicore::nn::Parameter GateUpParallelLinear::get_up_weight_awq() const {
    return infinicore::nn::Parameter(weight_->narrow({{1, weight_->size(1) / 2, weight_->size(1) / 2}}), 1, tp_rank_, tp_size_);
}

infinicore::nn::Parameter GateUpParallelLinear::get_gate_weight_scale_awq() const {
    return infinicore::nn::Parameter(weight_scale_->narrow({{1, 0, weight_scale_->size(1) / 2}}), 1, tp_rank_, tp_size_);
}

infinicore::nn::Parameter GateUpParallelLinear::get_up_weight_scale_awq() const {
    return infinicore::nn::Parameter(weight_scale_->narrow({{1, weight_scale_->size(1) / 2, weight_scale_->size(1) / 2}}), 1, tp_rank_, tp_size_);
}

infinicore::nn::Parameter GateUpParallelLinear::get_gate_weight_zeros_awq() const {
    return infinicore::nn::Parameter(weight_zeros_->narrow({{1, 0, weight_zeros_->size(1) / 2}}), 1, tp_rank_, tp_size_);
}

infinicore::nn::Parameter GateUpParallelLinear::get_up_weight_zeros_awq() const {
    return infinicore::nn::Parameter(weight_zeros_->narrow({{1, weight_zeros_->size(1) / 2, weight_zeros_->size(1) / 2}}), 1, tp_rank_, tp_size_);
}

infinicore::nn::Parameter GateUpParallelLinear::get_gate_g_idx_gptq() const {
    return infinicore::nn::Parameter(gidx_->narrow({{0, 0, gidx_->size(0)}}), 0, tp_rank_, tp_size_);
}

infinicore::nn::Parameter GateUpParallelLinear::get_up_g_idx_gptq() const {
    return infinicore::nn::Parameter(gidx_->narrow({{0, 0, gidx_->size(0)}}), 0, tp_rank_, tp_size_);
}

void QKVParallelLinear::register_parameters(std::function<void(const std::string &, infinicore::nn::Parameter)> register_fn,
                                            const std::string &q_name,
                                            const std::string &k_name,
                                            const std::string &v_name) {
    using namespace infinicore::quantization;
    auto scheme = this->get_quantization()->get_quant_scheme();
    switch (scheme) {
    case QuantScheme::NONE: {
        register_fn(q_name + ".weight", this->get_q_weight());
        register_fn(k_name + ".weight", this->get_k_weight());
        register_fn(v_name + ".weight", this->get_v_weight());
        break;
    }
    case QuantScheme::COMPRESSED_TENSOR_W8A8I8: {
        register_fn(q_name + ".weight", this->get_q_weight());
        register_fn(q_name + ".weight_scale", this->get_q_weight_scale());
        register_fn(k_name + ".weight", this->get_k_weight());
        register_fn(k_name + ".weight_scale", this->get_k_weight_scale());
        register_fn(v_name + ".weight", this->get_v_weight());
        register_fn(v_name + ".weight_scale", this->get_v_weight_scale());
        break;
    }
    case QuantScheme::AWQ_W4A16: {
        auto awq = std::static_pointer_cast<AWQ>(this->get_quantization());
        int packing_num = awq->get_packing_num();
        register_fn(q_name + ".qweight", this->get_q_weight_awq(packing_num));
        register_fn(q_name + ".qzeros", this->get_q_weight_zeros_awq(packing_num));
        register_fn(q_name + ".scales", this->get_q_weight_scale_awq(1));
        register_fn(k_name + ".qweight", this->get_k_weight_awq(packing_num));
        register_fn(k_name + ".qzeros", this->get_k_weight_zeros_awq(packing_num));
        register_fn(k_name + ".scales", this->get_k_weight_scale_awq(1));
        register_fn(v_name + ".qweight", this->get_v_weight_awq(packing_num));
        register_fn(v_name + ".qzeros", this->get_v_weight_zeros_awq(packing_num));
        register_fn(v_name + ".scales", this->get_v_weight_scale_awq(1));
        break;
    }
    case QuantScheme::GPTQ_W4A16:
    case QuantScheme::GPTQ_W4A16_QY: {
        register_fn(q_name + ".qweight", this->get_q_weight_awq(1));
        register_fn(q_name + ".qzeros", this->get_q_weight_zeros_awq(8));
        register_fn(q_name + ".scales", this->get_q_weight_scale_awq(1));
        register_fn(q_name + ".g_idx", this->get_q_g_idx_gptq());
        register_fn(k_name + ".qweight", this->get_k_weight_awq(1));
        register_fn(k_name + ".qzeros", this->get_k_weight_zeros_awq(8));
        register_fn(k_name + ".scales", this->get_k_weight_scale_awq(1));
        register_fn(k_name + ".g_idx", this->get_k_g_idx_gptq());
        register_fn(v_name + ".qweight", this->get_v_weight_awq(1));
        register_fn(v_name + ".qzeros", this->get_v_weight_zeros_awq(8));
        register_fn(v_name + ".scales", this->get_v_weight_scale_awq(1));
        register_fn(v_name + ".g_idx", this->get_v_g_idx_gptq());
        break;
    }
    default:
        throw std::runtime_error("QKVParallelLinear: unsupported quantization scheme");
    }
    if (this->has_q_bias()) {
        register_fn(q_name + ".bias", this->get_q_bias());
    }
    if (this->has_k_bias()) {
        register_fn(k_name + ".bias", this->get_k_bias());
    }
    if (this->has_v_bias()) {
        register_fn(v_name + ".bias", this->get_v_bias());
    }
}

void GateUpParallelLinear::register_parameters(std::function<void(const std::string &, infinicore::nn::Parameter)> register_fn,
                                               const std::string &gate_name,
                                               const std::string &up_name) {
    using namespace infinicore::quantization;
    auto scheme = this->get_quantization()->get_quant_scheme();
    switch (scheme) {
    case QuantScheme::NONE: {
        register_fn(gate_name + ".weight", this->get_gate_weight());
        register_fn(up_name + ".weight", this->get_up_weight());
        break;
    }
    case QuantScheme::COMPRESSED_TENSOR_W8A8I8: {
        register_fn(gate_name + ".weight", this->get_gate_weight());
        register_fn(gate_name + ".weight_scale", this->get_gate_weight_scale());
        register_fn(up_name + ".weight", this->get_up_weight());
        register_fn(up_name + ".weight_scale", this->get_up_weight_scale());
        break;
    }
    case QuantScheme::AWQ_W4A16: {
        register_fn(gate_name + ".qweight", this->get_gate_weight_awq());
        register_fn(gate_name + ".qzeros", this->get_gate_weight_zeros_awq());
        register_fn(gate_name + ".scales", this->get_gate_weight_scale_awq());
        register_fn(up_name + ".qweight", this->get_up_weight_awq());
        register_fn(up_name + ".qzeros", this->get_up_weight_zeros_awq());
        register_fn(up_name + ".scales", this->get_up_weight_scale_awq());
        break;
    }
    case QuantScheme::GPTQ_W4A16:
    case QuantScheme::GPTQ_W4A16_QY: {
        register_fn(gate_name + ".qweight", this->get_gate_weight_awq());
        register_fn(gate_name + ".qzeros", this->get_gate_weight_zeros_awq());
        register_fn(gate_name + ".scales", this->get_gate_weight_scale_awq());
        register_fn(gate_name + ".g_idx", this->get_gate_g_idx_gptq());
        register_fn(up_name + ".qweight", this->get_up_weight_awq());
        register_fn(up_name + ".qzeros", this->get_up_weight_zeros_awq());
        register_fn(up_name + ".scales", this->get_up_weight_scale_awq());
        register_fn(up_name + ".g_idx", this->get_up_g_idx_gptq());
        break;
    }
    default:
        throw std::runtime_error("GateUpParallelLinear: unsupported quantization scheme");
    }
    if (this->has_gate_bias()) {
        register_fn(gate_name + ".bias", this->get_gate_bias());
    }
    if (this->has_up_bias()) {
        register_fn(up_name + ".bias", this->get_up_bias());
    }
}

} // namespace infinilm::layers::linear
