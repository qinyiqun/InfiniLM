#include "../csrc/layers/quantization/weight_preprocessor.hpp"
#include <iostream>
#include <vector>
#include <cstring>
#include <cassert>
#include <iomanip>

// Simple test for GPTQ preprocessing logic

void test_param_int32_to_fp16_weights() {
    std::cout << "Testing param_int32_to_fp16_weights..." << std::endl;

    // Create test data for qzeros (int32)
    std::vector<int32_t> qzeros_data = {
        0x12345678, 0x9ABCDEF0, 0x11111111, 0x22222222
    };
    std::vector<size_t> qzeros_shape = {4, 2}; // 4x2 tensor

    auto qzeros_tensor = infinicore::Tensor::from_data(
        qzeros_data.data(),
        infinicore::DataType::I32,
        qzeros_shape
    );

    // Create preprocessor instance
    infinilm::quantization::GPTQWeightPreprocessor preprocessor;

    // Test the preprocessing
    auto result = preprocessor.param_int32_to_fp16_weights(qzeros_tensor, 4);

    std::cout << "Original qzeros shape: [";
    for (size_t i = 0; i < qzeros_shape.size(); ++i) {
        std::cout << qzeros_shape[i];
        if (i < qzeros_shape.size() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;

    std::cout << "Result shape: [";
    auto result_shape = result.shape();
    for (size_t i = 0; i < result_shape.size(); ++i) {
        std::cout << result_shape[i];
        if (i < result_shape.size() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;

    std::cout << "Result dtype: " << static_cast<int>(result.dtype()) << std::endl;

    // Expected shape should be [4, 2, 8] (added dimension of 32/4 = 8)
    // Expected dtype should be F16 (2)

    assert(result_shape.size() == 3 && "Result should have 3 dimensions");
    assert(result_shape[0] == 4 && result_shape[1] == 2 && result_shape[2] == 8 && "Result shape should be [4, 2, 8]");
    assert(result.dtype() == infinicore::DataType::F16 && "Result dtype should be F16");

    std::cout << "✓ param_int32_to_fp16_weights test passed" << std::endl;
}

void test_param_int32_to_uint8_weights() {
    std::cout << "Testing param_int32_to_uint8_weights..." << std::endl;

    // Create test data for qweight (int32)
    std::vector<int32_t> qweight_data = {
        0x12345678, 0x9ABCDEF0, 0x11111111, 0x22222222
    };
    std::vector<size_t> qweight_shape = {4, 2}; // 4x2 tensor

    auto qweight_tensor = infinicore::Tensor::from_data(
        qweight_data.data(),
        infinicore::DataType::I32,
        qweight_shape
    );

    // Create preprocessor instance
    infinilm::quantization::GPTQWeightPreprocessor preprocessor;

    // Test the preprocessing
    auto result = preprocessor.param_int32_to_uint8_weights(qweight_tensor, 4);

    std::cout << "Original qweight shape: [";
    for (size_t i = 0; i < qweight_shape.size(); ++i) {
        std::cout << qweight_shape[i];
        if (i < qweight_shape.size() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;

    std::cout << "Result shape: [";
    auto result_shape = result.shape();
    for (size_t i = 0; i < result_shape.size(); ++i) {
        std::cout << result_shape[i];
        if (i < result_shape.size() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;

    std::cout << "Result dtype: " << static_cast<int>(result.dtype()) << std::endl;

    // Expected shape should be [4, 16] (last dim multiplied by 32/4 = 8)
    // Expected dtype should be U8 (1)

    assert(result_shape.size() == 2 && "Result should have 2 dimensions");
    assert(result_shape[0] == 4 && result_shape[1] == 16 && "Result shape should be [4, 16]");
    assert(result.dtype() == infinicore::DataType::U8 && "Result dtype should be U8");

    std::cout << "✓ param_int32_to_uint8_weights test passed" << std::endl;
}

void test_combine_low_nibbles() {
    std::cout << "Testing combine_low_nibbles..." << std::endl;

    // Create test data - pairs of bytes where we need to combine low nibbles
    std::vector<uint8_t> test_data = {
        0xF1, 0xE2,  // Should produce 0x2F (E | 1)
        0x34, 0x56,  // Should produce 0x65 (6 | 5)
        0x78, 0x9A,  // Should produce 0xAB (A | B)
        0x11, 0x22   // Should produce 0x21 (2 | 1)
    };
    std::vector<size_t> test_shape = {8}; // 8 elements

    auto test_tensor = infinicore::Tensor::from_data(
        test_data.data(),
        infinicore::DataType::U8,
        test_shape
    );

    // Create preprocessor instance
    infinilm::quantization::GPTQWeightPreprocessor preprocessor;

    // Test the preprocessing
    auto result = preprocessor.combine_low_nibbles(test_tensor);

    std::cout << "Original shape: [";
    auto original_shape = test_tensor.shape();
    for (size_t i = 0; i < original_shape.size(); ++i) {
        std::cout << original_shape[i];
        if (i < original_shape.size() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;

    std::cout << "Original data: ";
    auto* original_data = static_cast<const uint8_t*>(test_tensor.data());
    for (size_t i = 0; i < std::min(size_t(8), test_tensor.numel()); ++i) {
        std::cout << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(original_data[i]) << " ";
    }
    std::cout << std::dec << std::endl;

    std::cout << "Result shape: [";
    auto result_shape = result.shape();
    for (size_t i = 0; i < result_shape.size(); ++i) {
        std::cout << result_shape[i];
        if (i < result_shape.size() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;

    std::cout << "Result data: ";
    auto* result_data = static_cast<const uint8_t*>(result.data());
    for (size_t i = 0; i < result.numel(); ++i) {
        std::cout << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(result_data[i]) << " ";
    }
    std::cout << std::dec << std::endl;

    // Expected shape should be [4] (half the elements)
    // Expected results: 0x2F, 0x65, 0xAB, 0x21

    assert(result_shape.size() == 1 && "Result should have 1 dimension");
    assert(result_shape[0] == 4 && "Result shape should be [4]");
    assert(result.dtype() == infinicore::DataType::U8 && "Result dtype should be U8");

    // Verify the actual data values
    auto result_data_ptr = static_cast<const uint8_t*>(result.data());
    assert(result_data_ptr[0] == 0x2F && "First pair should produce 0x2F");
    assert(result_data_ptr[1] == 0x65 && "Second pair should produce 0x65");
    assert(result_data_ptr[2] == 0xAB && "Third pair should produce 0xAB");
    assert(result_data_ptr[3] == 0x21 && "Fourth pair should produce 0x21");

    std::cout << "✓ combine_low_nibbles test passed" << std::endl;
}

int main() {
    std::cout << "=== GPTQ Weight Preprocessing Tests ===" << std::endl;

    try {
        test_param_int32_to_fp16_weights();
        std::cout << std::endl;

        test_param_int32_to_uint8_weights();
        std::cout << std::endl;

        test_combine_low_nibbles();
        std::cout << std::endl;

        std::cout << "=== All tests passed! ===" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "✗ Test failed with exception: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}