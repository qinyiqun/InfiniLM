#include "gtest/gtest.h"
#include "../csrc/layers/quantization/weight_preprocessor.hpp"
#include "../csrc/config/model_config.hpp"
#include "infinicore/tensor.hpp"

using namespace infinilm::quantization;
using namespace infinilm::config;

class WeightPreprocessorTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create a basic model config for testing
        nlohmann::json config_json = {
            {"model_type", "llama"},
            {"hidden_size", 4096},
            {"num_attention_heads", 32},
            {"num_hidden_layers", 32},
            {"vocab_size", 32000},
            {"quantization", nullptr}  // No quantization by default
        };

        model_config_ = std::make_shared<ModelConfig>(config_json);
    }

    std::shared_ptr<ModelConfig> model_config_;
};

// Test NoOpWeightPreprocessor
TEST_F(WeightPreprocessorTest, NoOpPreprocessorShouldNotProcess) {
    NoOpWeightPreprocessor preprocessor;

    // Should not process any parameters
    EXPECT_FALSE(preprocessor.should_process("model.layers.0.self_attn.q_proj.weight"));
    EXPECT_FALSE(preprocessor.should_process("model.embed_tokens.weight"));

    // Should return NONE quant scheme
    EXPECT_EQ(preprocessor.get_quant_scheme(), infinicore::quantization::QuantScheme::NONE);
}

// Test GPTQWeightPreprocessor
TEST_F(WeightPreprocessorTest, GPTQPreprocessorShouldProcessQuantizedWeights) {
    GPTQWeightPreprocessor preprocessor;

    // Should process quantized weights
    EXPECT_TRUE(preprocessor.should_process("model.layers.0.self_attn.q_proj.qweight"));
    EXPECT_TRUE(preprocessor.should_process("model.layers.0.self_attn.q_proj.qzeros"));
    EXPECT_TRUE(preprocessor.should_process("model.layers.0.self_attn.q_proj.scales"));
    EXPECT_TRUE(preprocessor.should_process("model.layers.0.self_attn.q_proj.g_idx"));

    // Should not process regular weights
    EXPECT_FALSE(preprocessor.should_process("model.layers.0.self_attn.q_proj.weight"));
    EXPECT_FALSE(preprocessor.should_process("model.embed_tokens.weight"));

    // Should return GPTQ quant scheme
    EXPECT_EQ(preprocessor.get_quant_scheme(), infinicore::quantization::QuantScheme::GPTQ_W4A16_QY);
}

// Test AWQWeightPreprocessor
TEST_F(WeightPreprocessorTest, AWQPreprocessorShouldProcessQuantizedWeights) {
    AWQWeightPreprocessor preprocessor;

    // Should process quantized weights
    EXPECT_TRUE(preprocessor.should_process("model.layers.0.self_attn.q_proj.qweight"));
    EXPECT_TRUE(preprocessor.should_process("model.layers.0.self_attn.q_proj.qzeros"));
    EXPECT_TRUE(preprocessor.should_process("model.layers.0.self_attn.q_proj.scales"));

    // Should not process regular weights or g_idx
    EXPECT_FALSE(preprocessor.should_process("model.layers.0.self_attn.q_proj.weight"));
    EXPECT_FALSE(preprocessor.should_process("model.layers.0.self_attn.q_proj.g_idx"));

    // Should return AWQ quant scheme
    EXPECT_EQ(preprocessor.get_quant_scheme(), infinicore::quantization::QuantScheme::AWQ_W4A16);
}

// Test W8A8WeightPreprocessor
TEST_F(WeightPreprocessorTest, W8A8PreprocessorShouldProcessCompressedWeights) {
    W8A8WeightPreprocessor preprocessor;

    // Should process weights and scales
    EXPECT_TRUE(preprocessor.should_process("model.layers.0.self_attn.q_proj.weight"));
    EXPECT_TRUE(preprocessor.should_process("model.layers.0.self_attn.q_proj.weight_scale"));

    // Should not process qweight/qzeros (those are for 4-bit)
    EXPECT_FALSE(preprocessor.should_process("model.layers.0.self_attn.q_proj.qweight"));
    EXPECT_FALSE(preprocessor.should_process("model.layers.0.self_attn.q_proj.qzeros"));

    // Should return W8A8 quant scheme
    EXPECT_EQ(preprocessor.get_quant_scheme(), infinicore::quantization::QuantScheme::COMPRESSED_TENSOR_W8A8I8);
}

// Test WeightPreprocessorFactory
TEST_F(WeightPreprocessorTest, FactoryShouldCreateCorrectPreprocessor) {
    // Test NONE scheme
    auto none_preprocessor = WeightPreprocessorFactory::create(infinicore::quantization::QuantScheme::NONE);
    EXPECT_NE(none_preprocessor, nullptr);
    EXPECT_EQ(none_preprocessor->get_quant_scheme(), infinicore::quantization::QuantScheme::NONE);

    // Test GPTQ scheme
    auto gptq_preprocessor = WeightPreprocessorFactory::create(infinicore::quantization::QuantScheme::GPTQ_W4A16_QY);
    EXPECT_NE(gptq_preprocessor, nullptr);
    EXPECT_EQ(gptq_preprocessor->get_quant_scheme(), infinicore::quantization::QuantScheme::GPTQ_W4A16_QY);

    // Test AWQ scheme
    auto awq_preprocessor = WeightPreprocessorFactory::create(infinicore::quantization::QuantScheme::AWQ_W4A16);
    EXPECT_NE(awq_preprocessor, nullptr);
    EXPECT_EQ(awq_preprocessor->get_quant_scheme(), infinicore::quantization::QuantScheme::AWQ_W4A16);

    // Test W8A8 scheme
    auto w8a8_preprocessor = WeightPreprocessorFactory::create(infinicore::quantization::QuantScheme::COMPRESSED_TENSOR_W8A8I8);
    EXPECT_NE(w8a8_preprocessor, nullptr);
    EXPECT_EQ(w8a8_preprocessor->get_quant_scheme(), infinicore::quantization::QuantScheme::COMPRESSED_TENSOR_W8A8I8);
}

// Test parameter name patterns
TEST_F(WeightPreprocessorTest, ParameterNamePatterns) {
    GPTQWeightPreprocessor preprocessor;

    // Test various parameter name patterns
    EXPECT_TRUE(preprocessor.should_process("model.layers.0.mlp.gate_proj.qweight"));
    EXPECT_TRUE(preprocessor.should_process("model.layers.31.mlp.up_proj.qzeros"));
    EXPECT_TRUE(preprocessor.should_process("model.layers.15.mlp.down_proj.scales"));
    EXPECT_TRUE(preprocessor.should_process("model.layers.5.mlp.down_proj.g_idx"));

    // Test edge cases
    EXPECT_FALSE(preprocessor.should_process(""));  // Empty string
    EXPECT_FALSE(preprocessor.should_process("qweight"));  // No prefix
    EXPECT_FALSE(preprocessor.should_process("model.layers.0.weight"));  // Not quantized
}

// Test preprocessing with different quantization schemes
TEST_F(WeightPreprocessorTest, PreprocessWithDifferentSchemes) {
    // Create a simple tensor for testing
    auto tensor = infinicore::Tensor::ones({4096, 4096}, infinicore::DataType::F32);

    // Test GPTQ preprocessing
    auto gptq_preprocessor = WeightPreprocessorFactory::create(infinicore::quantization::QuantScheme::GPTQ_W4A16_QY);
    auto gptq_result = gptq_preprocessor->preprocess_weight("model.layers.0.q_proj.qweight", tensor, model_config_);
    EXPECT_NE(gptq_result, nullptr);  // Should return a valid tensor

    // Test AWQ preprocessing
    auto awq_preprocessor = WeightPreprocessorFactory::create(infinicore::quantization::QuantScheme::AWQ_W4A16);
    auto awq_result = awq_preprocessor->preprocess_weight("model.layers.0.q_proj.qweight", tensor, model_config_);
    EXPECT_NE(awq_result, nullptr);  // Should return a valid tensor

    // Test W8A8 preprocessing
    auto w8a8_preprocessor = WeightPreprocessorFactory::create(infinicore::quantization::QuantScheme::COMPRESSED_TENSOR_W8A8I8);
    auto w8a8_result = w8a8_preprocessor->preprocess_weight("model.layers.0.q_proj.weight", tensor, model_config_);
    EXPECT_NE(w8a8_result, nullptr);  // Should return a valid tensor
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}