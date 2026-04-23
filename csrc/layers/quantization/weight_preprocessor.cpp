// #include "weight_preprocessor.hpp"
// #include "infinicore/ops.hpp"
// #include <spdlog/spdlog.h>
// #include <regex>
// #include <vector>
// #include <algorithm>
// #include <cstring>

// namespace infinilm::quantization {

// // ============================================================
// // WeightPreprocessorFactory Implementation
// // ============================================================

// std::unique_ptr<BaseWeightPreprocessor> WeightPreprocessorFactory::create(
//     infinicore::quantization::QuantScheme quant_scheme) {

//     switch (quant_scheme) {
//         case infinicore::quantization::QuantScheme::GPTQ_W4A16_QY:
//             return std::make_unique<GPTQWeightPreprocessor>();

//         case infinicore::quantization::QuantScheme::AWQ_W4A16:
//             return std::make_unique<AWQWeightPreprocessor>();

//         case infinicore::quantization::QuantScheme::COMPRESSED_TENSOR_W8A8I8:
//             return std::make_unique<W8A8WeightPreprocessor>();

//         case infinicore::quantization::QuantScheme::NONE:
//         default:
//             return std::make_unique<NoOpWeightPreprocessor>();
//     }
// }

// // ============================================================
// // GPTQWeightPreprocessor Implementation
// // ============================================================

// bool GPTQWeightPreprocessor::should_process(const std::string& param_name) const {
//     // Process quantized weights (qweight) and related parameters
//     static const std::regex qweight_pattern(R"(.*\.qweight$)");
//     static const std::regex qzeros_pattern(R"(.*\.qzeros$)");
//     static const std::regex scales_pattern(R"(.*\.scales$)");
//     static const std::regex gidx_pattern(R"(.*\.g_idx$)");

//     return std::regex_search(param_name, qweight_pattern) ||
//            std::regex_search(param_name, qzeros_pattern) ||
//            std::regex_search(param_name, scales_pattern) ||
//            std::regex_search(param_name, gidx_pattern);
// }

// infinicore::Tensor GPTQWeightPreprocessor::param_int32_to_fp16_weights(
//     const infinicore::Tensor& qzeros,
//     int bits) {

//     // Get CPU data from input tensor
//     auto qzeros_cpu = qzeros.to(infinicore::Device::cpu());
//     auto* qzeros_data = static_cast<const int32_t*>(qzeros_cpu.data());

//     // Get shape information
//     auto qzeros_shape = qzeros_cpu.shape();
//     size_t total_elements = 1;
//     for (auto dim : qzeros_shape) {
//         total_elements *= dim;
//     }

//     // Calculate number of elements after adding extra dimension
//     size_t elements_per_channel = total_elements / qzeros_shape.back();
//     size_t extra_dim = 32 / bits;

//     // Create output buffer for float16 data
//     std::vector<uint16_t> output_data(total_elements * extra_dim);

//     // Process each element: add 1, bitwise AND, convert to float16
//     int32_t mask = (1 << bits) - 1;
//     for (size_t i = 0; i < total_elements; ++i) {
//         for (size_t j = 0; j < extra_dim; ++j) {
//             size_t output_idx = i * extra_dim + j;

//             // Right shift by position
//             int32_t shifted = (qzeros_data[i] >> j) & 0xFF;

//             // Add 1
//             shifted = shifted + 1;

//             // Bitwise AND with mask
//             int32_t masked = shifted & mask;

//             // Convert to float16 (assuming the result fits in 16 bits)
//             output_data[output_idx] = static_cast<uint16_t>(masked);
//         }
//     }

//     // Create output tensor with new shape
//     std::vector<size_t> output_shape = qzeros_shape;
//     output_shape.push_back(extra_dim);

//     return infinicore::Tensor::from_data(
//         output_data.data(),
//         infinicore::DataType::F16,
//         output_shape
//     ).to(qzeros.device());
// }

// infinicore::Tensor GPTQWeightPreprocessor::param_int32_to_uint8_weights(
//     const infinicore::Tensor& weight,
//     int bits) {

//     // Get CPU data from input tensor
//     auto weight_cpu = weight.to(infinicore::Device::cpu());
//     auto* weight_data = static_cast<const int32_t*>(weight_cpu.data());

//     // Get shape information
//     auto weight_shape = weight_cpu.shape();
//     size_t total_elements = 1;
//     for (auto dim : weight_shape) {
//         total_elements *= dim;
//     }

//     // Calculate new shape after processing
//     size_t new_shape_last_dim = weight_shape.back() * (32 / bits);
//     std::vector<size_t> new_shape = weight_shape;
//     new_shape.back() = new_shape_last_dim;

//     // Create output buffer for uint8 data
//     std::vector<uint8_t> output_data(total_elements * (32 / bits));

//     // Process each element: right shift by position, bitwise AND
//     int32_t mask = (1 << bits) - 1;
//     for (size_t i = 0; i < total_elements; ++i) {
//         for (size_t j = 0; j < 32 / bits; ++j) {
//             size_t output_idx = i * (32 / bits) + j;

//             // Right shift by position
//             int32_t shifted = (weight_data[i] >> (j * bits)) & 0xFF;

//             // Bitwise AND with mask
//             int32_t masked = shifted & mask;

//             output_data[output_idx] = static_cast<uint8_t>(masked);
//         }
//     }

//     // Create output tensor
//     return infinicore::Tensor::from_data(
//         output_data.data(),
//         infinicore::DataType::U8,
//         new_shape
//     ).to(weight.device());
// }

// infinicore::Tensor GPTQWeightPreprocessor::combine_low_nibbles(const infinicore::Tensor& arr) {
//     // This function combines low 4 bits from pairs of int8 elements
//     // Similar to Python's combine_low_nibbles function

//     // Get CPU data
//     auto arr_cpu = arr.to(infinicore::Device::cpu());
//     auto* data = static_cast<const uint8_t*>(arr_cpu.data());

//     // Get total number of elements
//     size_t total_elements = arr_cpu.numel();

//     // Ensure even number of elements
//     if (total_elements % 2 != 0) {
//         spdlog::warn("combine_low_nibbles: array length must be even, got {}", total_elements);
//         return arr; // Return original if odd length
//     }

//     // Create output array (half the size)
//     std::vector<uint8_t> combined_data(total_elements / 2);

//     // Extract low 4 bits from each pair and combine
//     for (size_t i = 0; i < total_elements; i += 2) {
//         uint8_t low_nibble0 = data[i] & 0x0F;     // Low 4 bits of first element
//         uint8_t low_nibble1 = data[i + 1] & 0x0F;  // Low 4 bits of second element

//         // Combine: high nibble from second element, low nibble from first element
//         combined_data[i / 2] = (low_nibble1 << 4) | low_nibble0;
//     }

//     // Create output tensor with new shape
//     auto arr_shape = arr_cpu.shape();
//     arr_shape.back() = arr_shape.back() / 2;

//     return infinicore::Tensor::from_data(
//         combined_data.data(),
//         infinicore::DataType::U8,
//         arr_shape
//     ).to(arr.device());
// }

// infinicore::Tensor GPTQWeightPreprocessor::preprocess_weight(
//     const std::string& param_name,
//     const infinicore::Tensor& tensor,
//     const std::shared_ptr<infinilm::config::ModelConfig>& model_config) {

//     // For GPTQ, we need to process qweights, qzeros, and scales
//     // Following the Python implementation logic

//     spdlog::debug("Preprocessing GPTQ weight: {}", param_name);

//     // Handle qweight parameter
//     if (param_name.find(".qweight") != std::string::npos) {
//         // Convert int32 qweight to uint8 format
//         auto qweight_uint8 = param_int32_to_uint8_weights(tensor, 4);

//         // For 4-bit weights, apply combine_low_nibbles
//         auto qweight_combined = combine_low_nibbles(qweight_uint8);

//         spdlog::debug("Processed qweight: {} -> shape: {}, dtype: U8",
//                      param_name, qweight_combined.shape());

//         return qweight_combined;
//     }

//     // Handle qzeros parameter
//     if (param_name.find(".qzeros") != std::string::npos) {
//         // Convert int32 qzeros to fp16 format
//         auto qzeros_fp16 = param_int32_to_fp16_weights(tensor, 4);

//         spdlog::debug("Processed qzeros: {} -> shape: {}, dtype: F16",
//                      param_name, qzeros_fp16.shape());

//         return qzeros_fp16;
//     }

//     // Handle scales parameter
//     if (param_name.find(".scales") != std::string::npos) {
//         // Convert scales to float16 (if not already)
//         if (tensor.dtype() != infinicore::DataType::F16) {
//             auto scales_fp16 = tensor.to(infinicore::DataType::F16).to(tensor.device());

//             spdlog::debug("Processed scales: {} -> shape: {}, dtype: F16",
//                          param_name, scales_fp16.shape());

//             return scales_fp16;
//         } else {
//             spdlog::debug("Scales already in F16 format: {}", param_name);
//             return tensor;
//         }
//     }

//     // Handle g_idx (activation ordering) - no transformation needed
//     if (param_name.find(".g_idx") != std::string::npos) {
//         spdlog::debug("No transformation needed for g_idx: {}", param_name);
//         return tensor;
//     }

//     return tensor;
// }

// // ============================================================
// // AWQWeightPreprocessor Implementation
// // ============================================================

// bool AWQWeightPreprocessor::should_process(const std::string& param_name) const {
//     // Process AWQ-specific parameters
//     static const std::regex qweight_pattern(R"(.*\.qweight$)");
//     static const std::regex qzeros_pattern(R"(.*\.qzeros$)");
//     static const std::regex scales_pattern(R"(.*\.scales$)");

//     return std::regex_search(param_name, qweight_pattern) ||
//            std::regex_search(param_name, qzeros_pattern) ||
//            std::regex_search(param_name, scales_pattern);
// }

// infinicore::Tensor AWQWeightPreprocessor::preprocess_weight(
//     const std::string& param_name,
//     const infinicore::Tensor& tensor,
//     const std::shared_ptr<infinilm::config::ModelConfig>& model_config) {

//     // For AWQ, we typically need to:
//     // 1. Optimize packed weight layout
//     // 2. Pre-compute scaled zeros for efficient inference
//     // 3. Handle per-channel scaling optimization

//     spdlog::debug("Preprocessing AWQ weight: {}", param_name);

//     // Check if this is a qweight parameter
//     if (param_name.find(".qweight") != std::string::npos) {
//         // AWQ qweights are packed similarly to GPTQ
//         // Return as-is - can be enhanced with specific optimizations
//         return tensor;
//     }

//     // Handle qzeros - AWQ might have specific zero point handling
//     if (param_name.find(".qzeros") != std::string::npos) {
//         // AWQ zeros might need to be combined with scales
//         // For now, return as-is - can be enhanced with specific transforms
//         return tensor;
//     }

//     // Handle scales
//     if (param_name.find(".scales") != std::string::npos) {
//         // AWQ scales might benefit from pre-normalization
//         // For now, return as-is - can be enhanced with specific optimizations
//         return tensor;
//     }

//     return tensor;
// }

// // ============================================================
// // W8A8WeightPreprocessor Implementation
// // ============================================================

// bool W8A8WeightPreprocessor::should_process(const std::string& param_name) const {
//     // Process W8A8 quantized weights and scales
//     static const std::regex weight_pattern(R"(.*\.weight$)");
//     static const std::regex weight_scale_pattern(R"(.*\.weight_scale$)");

//     return std::regex_search(param_name, weight_pattern) ||
//            std::regex_search(param_name, weight_scale_pattern);
// }

// infinicore::Tensor W8A8WeightPreprocessor::preprocess_weight(
//     const std::string& param_name,
//     const infinicore::Tensor& tensor,
//     const std::shared_ptr<infinilm::config::ModelConfig>& model_config) {

//     // For W8A8, we typically need to:
//     // 1. Ensure proper int8 weight layout
//     // 2. Optimize scale factors (per-tensor vs per-channel)
//     // 3. Potentially pre-compute scale inversion for dequantization

//     spdlog::debug("Preprocessing W8A8 weight: {}", param_name);

//     // Handle weight scales
//     if (param_name.find(".weight_scale") != std::string::npos) {
//         // Might want to pre-compute inverse scales for faster dequantization
//         // For now, return as-is - can be enhanced with specific optimizations
//         return tensor;
//     }

//     // Handle int8 weights
//     if (param_name.find(".weight") != std::string::npos &&
//         param_name.find(".weight_scale") == std::string::npos) {
//         // Ensure proper int8 layout
//         // For now, return as-is - can be enhanced with layout optimization
//         return tensor;
//     }

//     return tensor;
// }

// // ============================================================
// // Utility Functions
// // ============================================================

// void preprocess_model_weights(
//     infinicore::nn::Module* model,
//     const std::shared_ptr<infinilm::config::ModelConfig>& model_config) {

//     if (!model || !model_config) {
//         spdlog::warn("preprocess_model_weights: model or model_config is null");
//         return;
//     }

//     // Get the quantization scheme
//     auto quant_scheme = model_config->get_quant_scheme();

//     // Create appropriate preprocessor
//     auto preprocessor = WeightPreprocessorFactory::create(quant_scheme);

//     if (!preprocessor) {
//         spdlog::debug("No weight preprocessor for quantization scheme: {}",
//                      static_cast<int>(quant_scheme));
//         return;
//     }

//     spdlog::info("Starting weight preprocessing for quantization scheme: {}",
//                 static_cast<int>(quant_scheme));

//     // Get all named parameters from the model
//     auto named_parameters = model->named_parameters();

//     int processed_count = 0;
//     for (const auto& [name, param] : named_parameters) {
//         // Check if this parameter should be processed
//         if (!preprocessor->should_process(name)) {
//             continue;
//         }

//         try {
//             // Get the tensor from the parameter
//             auto tensor = param.tensor();

//             // Preprocess the weight
//             auto preprocessed_tensor = preprocessor->preprocess_weight(name, tensor, model_config);

//             // Update the parameter with the preprocessed tensor
//             if (preprocessed_tensor != tensor) {
//                 param.set_tensor(preprocessed_tensor);
//                 spdlog::debug("Preprocessed weight: {}", name);
//                 processed_count++;
//             }

//         } catch (const std::exception& e) {
//             spdlog::error("Failed to preprocess weight {}: {}", name, e.what());
//             // Continue with other weights even if one fails
//         }
//     }

//     spdlog::info("Weight preprocessing completed. Processed {} weights.", processed_count);
// }

// } // namespace infinilm::quantization