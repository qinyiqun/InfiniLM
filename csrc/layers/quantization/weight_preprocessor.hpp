// #pragma once

// #include "../../config/model_config.hpp"
// #include "infinicore/nn/module.hpp"
// #include "infinicore/tensor.hpp"
// #include <memory>
// #include <string>

// namespace infinilm::quantization {

// /**
//  * @brief Base class for weight preprocessing
//  *
//  * This class defines the interface for preprocessing weights after loading,
//  * similar to vLLM's process_weights_after_loading functionality.
//  *
//  * The preprocessing occurs after:
//  * 1. Model weights are loaded from checkpoint
//  * 2. Weight fusion (e.g., QKV fusion, Gate-Up fusion)
//  * 3. Tensor parallelism slicing
//  *
//  * Different quantization schemes can implement specific preprocessing logic.
//  */
// class BaseWeightPreprocessor {
// public:
//     virtual ~BaseWeightPreprocessor() = default;

//     /**
//      * @brief Preprocess weights for a specific parameter
//      *
//      * @param param_name Name of the parameter (e.g., "model.layers.0.self_attn.q_proj.weight")
//      * @param tensor The tensor to preprocess
//      * @param model_config Model configuration
//      * @return Preprocessed tensor (may be the same tensor if no preprocessing needed)
//      */
//     virtual infinicore::Tensor preprocess_weight(
//         const std::string &param_name,
//         const infinicore::Tensor &tensor,
//         const std::shared_ptr<infinilm::config::ModelConfig> &model_config) = 0;

//     /**
//      * @brief Check if this preprocessor should handle a given parameter
//      *
//      * @param param_name Name of the parameter
//      * @return true if this preprocessor should handle this parameter
//      */
//     virtual bool should_process(const std::string &param_name) const = 0;

//     /**
//      * @brief Get the quantization scheme this preprocessor is designed for
//      */
//     virtual infinicore::quantization::QuantScheme get_quant_scheme() const = 0;
// };

// /**
//  * @brief Factory for creating weight preprocessors based on quantization scheme
//  */
// class WeightPreprocessorFactory {
// public:
//     /**
//      * @brief Create a weight preprocessor for the given quantization scheme
//      *
//      * @param quant_scheme The quantization scheme
//      * @return Unique pointer to the appropriate preprocessor, or nullptr if no preprocessing needed
//      */
//     static std::unique_ptr<BaseWeightPreprocessor> create(
//         infinicore::quantization::QuantScheme quant_scheme);
// };

// /**
//  * @brief No-op preprocessor for non-quantized models
//  */
// class NoOpWeightPreprocessor : public BaseWeightPreprocessor {
// public:
//     infinicore::Tensor preprocess_weight(
//         const std::string &param_name,
//         const infinicore::Tensor &tensor,
//         const std::shared_ptr<infinilm::config::ModelConfig> &model_config) override {
//         return tensor; // No preprocessing needed
//     }

//     bool should_process(const std::string &param_name) const override {
//         return false; // Don't process any parameters
//     }

//     infinicore::quantization::QuantScheme get_quant_scheme() const override {
//         return infinicore::quantization::QuantScheme::NONE;
//     }
// };

// /**
//  * @brief GPTQ weight preprocessor for W4A16 quantization
//  *
//  * Handles preprocessing for GPTQ quantized weights, including:
//  * - Dequantization scaling factors
//  * - Zero point adjustments
//  * - Packing/unpacking optimization
//  */
// class GPTQWeightPreprocessor : public BaseWeightPreprocessor {
// public:
//     infinicore::Tensor preprocess_weight(
//         const std::string &param_name,
//         const infinicore::Tensor &tensor,
//         const std::shared_ptr<infinilm::config::ModelConfig> &model_config) override;

//     bool should_process(const std::string &param_name) const override;

//     infinicore::quantization::QuantScheme get_quant_scheme() const override {
//         return infinicore::quantization::QuantScheme::GPTQ_W4A16_QY;
//     }

// private:
//     /**
//      * @brief Convert int32 qzeros to fp16 format
//      *
//      * Similar to Python's param_int32_to_fp16_weights
//      * Performs bitwise operations to extract and convert zero points
//      */
//     infinicore::Tensor param_int32_to_fp16_weights(
//         const infinicore::Tensor &qzeros,
//         int bits = 4);

//     /**
//      * @brief Convert int32 qweight to uint8 format
//      *
//      * Similar to Python's param_int32_to_uint8_weights
//      * For 4-bit weights, also performs combine_low_nibbles operation
//      */
//     infinicore::Tensor param_int32_to_uint8_weights(
//         const infinicore::Tensor &weight,
//         int bits = 4);

//     /**
//      * @brief Combine low nibbles from adjacent int8 elements
//      *
//      * Similar to Python's combine_low_nibbles
//      * Takes low 4 bits from two consecutive int8 elements and combines them
//      */
//     infinicore::Tensor combine_low_nibbles(const infinicore::Tensor &arr);
// };

// /**
//  * @brief AWQ weight preprocessor for W4A16 quantization
//  *
//  * Handles preprocessing for AWQ quantized weights, including:
//  * - Activation-aware weight quantization scaling
//  * - Zero point and scale optimization
//  */
// class AWQWeightPreprocessor : public BaseWeightPreprocessor {
// public:
//     infinicore::Tensor preprocess_weight(
//         const std::string &param_name,
//         const infinicore::Tensor &tensor,
//         const std::shared_ptr<infinilm::config::ModelConfig> &model_config) override;

//     bool should_process(const std::string &param_name) const override;

//     infinicore::quantization::QuantScheme get_quant_scheme() const override {
//         return infinicore::quantization::QuantScheme::AWQ_W4A16;
//     }
// };

// /**
//  * @brief W8A8 compressed tensor weight preprocessor
//  *
//  * Handles preprocessing for W8A8 quantized weights, including:
//  * - Scale factor normalization
//  * - Per-channel/per-tensor scaling optimization
//  */
// class W8A8WeightPreprocessor : public BaseWeightPreprocessor {
// public:
//     infinicore::Tensor preprocess_weight(
//         const std::string &param_name,
//         const infinicore::Tensor &tensor,
//         const std::shared_ptr<infinilm::config::ModelConfig> &model_config) override;

//     bool should_process(const std::string &param_name) const override;

//     infinicore::quantization::QuantScheme get_quant_scheme() const override {
//         return infinicore::quantization::QuantScheme::COMPRESSED_TENSOR_W8A8I8;
//     }
// };

// /**
//  * @brief Utility function to preprocess all weights in a model
//  *
//  * This function iterates through all parameters in a model and applies
//  * the appropriate preprocessing based on the quantization scheme.
//  *
//  * @param model The model to preprocess
//  * @param model_config Model configuration
//  */
// void preprocess_model_weights(
//     infinicore::nn::Module *model,
//     const std::shared_ptr<infinilm::config::ModelConfig> &model_config);

// } // namespace infinilm::quantization