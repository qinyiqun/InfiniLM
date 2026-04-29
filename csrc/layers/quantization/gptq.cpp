#include "gptq.hpp"
#include "gptq_qy.hpp"
#include <spdlog/spdlog.h>

namespace infinilm::quantization {

std::vector<ParamDescriptor> GPTQ::get_param_layout(
    size_t in_features, size_t out_features,
    int split_dim, int tp_rank, int tp_size,
    int /*tp_num_heads*/,
    const infinicore::DataType &dtype,
    bool bias) const {

    int group_size = get_group_size();

    std::vector<ParamDescriptor> descs;
    descs.push_back({"qweight", {in_features / 8, out_features},
                     infinicore::DataType::I32, split_dim, tp_rank, tp_size});
    descs.push_back({"qzeros", {in_features / group_size, out_features / 8},
                     infinicore::DataType::I32, split_dim, tp_rank, tp_size});
    descs.push_back({"scales", {in_features / group_size, out_features},
                     dtype, split_dim, tp_rank, tp_size});
    descs.push_back({"g_idx", {in_features},
                     infinicore::DataType::I32, -1, tp_rank, tp_size});
    if (bias) {
        descs.push_back({"bias", {out_features}, dtype, -1, 0, 1});
    }
    return descs;
}

infinicore::Tensor GPTQ::forward(
    const ParamsMap &/*params*/,
    const infinicore::Tensor &/*input*/,
    bool /*has_bias*/) const {
    throw std::runtime_error("GPTQ_W4A16 must be converted to GPTQ_QY before forward pass. "
                             "Call process_weights_after_loading() first.");
}

void GPTQ::process_weights_after_loading(
    ParamsMap &params,
    const infinicore::Device &device) const {

    auto gptq_qy = std::make_shared<GPTQ_QY>(get_config());

    gptq_qy->convert_from_gptq_w4a16(
        params.at("qweight"), params.at("qzeros"), params.at("scales"),
        params.at("g_idx"), device);

    params["qweight"] = gptq_qy->get_converted_weight();
    params["qzeros"] = gptq_qy->get_converted_zeros();
    params["scales"] = gptq_qy->get_converted_scales();
    gptq_qy->release_buffers();
}

std::vector<SplitParam> GPTQ::split_params(
    const std::unordered_map<std::string, infinicore::nn::Parameter> &params,
    const std::vector<SplitInfo> &splits,
    int narrow_dim,
    int tp_rank, int tp_size, int tp_num_heads) const {

    std::vector<SplitParam> result;
    auto qw_it = params.find("qweight");
    auto qz_it = params.find("qzeros");
    auto sc_it = params.find("scales");
    auto gidx_it = params.find("g_idx");
    auto bias_it = params.find("bias");

    for (const auto &s : splits) {
        int num_heads = tp_num_heads > 0 ? tp_num_heads : 0;
        result.push_back({s.prefix + ".qweight",
                          infinicore::nn::Parameter(
                              qw_it->second->narrow({{static_cast<size_t>(narrow_dim), s.start, s.size}}),
                              narrow_dim, tp_rank, tp_size)});
        result.push_back({s.prefix + ".qzeros",
                          infinicore::nn::Parameter(
                              qz_it->second->narrow({{static_cast<size_t>(narrow_dim), s.start / 8, s.size / 8}}),
                              narrow_dim, tp_rank, tp_size)});
        result.push_back({s.prefix + ".scales",
                          infinicore::nn::Parameter(
                              sc_it->second->narrow({{static_cast<size_t>(narrow_dim), s.start, s.size}}),
                              narrow_dim, tp_rank, tp_size)});
        result.push_back({s.prefix + ".g_idx",
                          infinicore::nn::Parameter(
                              gidx_it->second->narrow({{0, 0, gidx_it->second->size(0)}}),
                              0, tp_rank, tp_size, num_heads)});
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
