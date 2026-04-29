#include "awq.hpp"
#include "infinicore/ops/linear_w4a16_awq.hpp"
#include <optional>

namespace infinilm::quantization {

std::vector<ParamDescriptor> AWQ::get_param_layout(
    size_t in_features, size_t out_features,
    int split_dim, int tp_rank, int tp_size,
    int /*tp_num_heads*/,
    const infinicore::DataType &dtype,
    bool bias) const {

    int group_size = get_group_size();
    int packing_num = get_packing_num();

    std::vector<ParamDescriptor> descs;
    descs.push_back({"qweight", {in_features, out_features / packing_num},
                     infinicore::DataType::I32, split_dim, tp_rank, tp_size});
    descs.push_back({"scales", {in_features / group_size, out_features},
                     dtype, split_dim, tp_rank, tp_size});
    descs.push_back({"qzeros", {in_features / group_size, out_features / packing_num},
                     infinicore::DataType::I32, split_dim, tp_rank, tp_size});
    if (bias) {
        descs.push_back({"bias", {out_features}, dtype, -1, 0, 1});
    }
    return descs;
}

infinicore::Tensor AWQ::forward(
    const ParamsMap &params,
    const infinicore::Tensor &input,
    bool has_bias) const {

    auto input_contiguous = input->is_contiguous() ? input : input->contiguous();
    auto qweight = params.at("qweight");
    auto scales = params.at("scales");
    auto qzeros = params.at("qzeros");

    std::optional<infinicore::Tensor> bias_opt;
    if (has_bias) {
        bias_opt = params.at("bias");
    }

    return infinicore::op::linear_w4a16_awq(input_contiguous->contiguous(), qweight, scales, qzeros, bias_opt);
}

std::vector<SplitParam> AWQ::split_params(
    const std::unordered_map<std::string, infinicore::nn::Parameter> &params,
    const std::vector<SplitInfo> &splits,
    int narrow_dim,
    int tp_rank, int tp_size, int tp_num_heads) const {

    int packing_num = get_packing_num();
    std::vector<SplitParam> result;
    auto qw_it = params.find("qweight");
    auto sc_it = params.find("scales");
    auto qz_it = params.find("qzeros");
    auto bias_it = params.find("bias");

    for (const auto &s : splits) {
        // qweight: narrow along narrow_dim, divide size by packing_num
        result.push_back({s.prefix + ".qweight",
                          infinicore::nn::Parameter(
                              qw_it->second->narrow({{static_cast<size_t>(narrow_dim), s.start / packing_num, s.size / packing_num}}),
                              narrow_dim, tp_rank, tp_size)});
        // scales: narrow along narrow_dim
        result.push_back({s.prefix + ".scales",
                          infinicore::nn::Parameter(
                              sc_it->second->narrow({{static_cast<size_t>(narrow_dim), s.start, s.size}}),
                              narrow_dim, tp_rank, tp_size)});
        // qzeros: narrow along narrow_dim, divide size by packing_num
        result.push_back({s.prefix + ".qzeros",
                          infinicore::nn::Parameter(
                              qz_it->second->narrow({{static_cast<size_t>(narrow_dim), s.start / packing_num, s.size / packing_num}}),
                              narrow_dim, tp_rank, tp_size)});
        if (bias_it != params.end()) {
            result.push_back({s.prefix + ".bias",
                              infinicore::nn::Parameter(
                                  bias_it->second->narrow({{0, s.start, s.size}}),
                                  0, tp_rank, tp_size)});
        }
    }
    return result;
}

} // namespace infinilm::quantization
