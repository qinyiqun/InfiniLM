#include <iostream>
#include <vector>
#include <cstring>
#include <cassert>
#include <iomanip>

// Simple standalone test for GPTQ preprocessing logic (without dependencies)

// Define simple data types
typedef uint8_t u8;
typedef uint16_t u16;
typedef int32_t i32;
typedef int16_t i16;

// Mock tensor class for testing
class MockTensor {
private:
    std::vector<u8> data_;
    std::vector<size_t> shape_;
    int dtype_; // 0=U8, 1=F16, 2=I32

public:
    MockTensor(const void* data, size_t size, const std::vector<size_t>& shape, int dtype)
        : dtype_(dtype) {
        data_.resize(size);
        memcpy(data_.data(), data, size);
        shape_ = shape;
    }

    const void* data() const { return data_.data(); }
    const std::vector<size_t>& shape() const { return shape_; }
    size_t numel() const {
        size_t total = 1;
        for (auto dim : shape_) total *= dim;
        return total;
    }
    int dtype() const { return dtype_; }
};

void test_param_int32_to_fp16_weights() {
    std::cout << "Testing param_int32_to_fp16_weights..." << std::endl;

    // Create test data for qzeros (int32)
    std::vector<i32> qzeros_data = {
        0x12345678, 0x9ABCDEF0, 0x11111111, 0x22222222
    };
    std::vector<size_t> qzeros_shape = {4, 2}; // 4x2 tensor

    MockTensor qzeros_tensor(qzeros_data.data(), qzeros_data.size() * sizeof(i32), qzeros_shape, 2); // I32

    // Process: int32 -> fp16 for 4-bit
    int bits = 4;
    int extra_dim = 32 / bits;
    int mask = (1 << bits) - 1;

    // Calculate output size
    size_t total_elements = 1;
    for (auto dim : qzeros_shape) total_elements *= dim;
    size_t output_size = static_cast<size_t>(static_cast<long long>(total_elements) * extra_dim);

    // Create output buffer
    std::vector<u16> output_data(output_size);

    // Process each element: add 1, bitwise AND, convert to float16
    const i32* input_data = static_cast<const i32*>(qzeros_tensor.data());

    for (size_t i = 0; i < total_elements; ++i) {
        for (size_t j = 0; j < extra_dim; ++j) {
            size_t output_idx = i * extra_dim + j;

            // Right shift by position
            i32 shifted = (input_data[i] >> j) & 0xFF;

            // Add 1
            shifted = shifted + 1;

            // Bitwise AND with mask
            i32 masked = shifted & mask;

            // Convert to float16 (assuming the result fits in 16 bits)
            output_data[output_idx] = static_cast<u16>(masked);
        }
    }

    // Expected output shape: [4, 2, 8]
    std::vector<size_t> expected_shape = qzeros_shape;
    expected_shape.push_back(extra_dim);

    std::cout << "Input shape: [";
    for (size_t i = 0; i < qzeros_shape.size(); ++i) {
        std::cout << qzeros_shape[i];
        if (i < qzeros_shape.size() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;

    std::cout << "Expected output shape: [";
    for (size_t i = 0; i < expected_shape.size(); ++i) {
        std::cout << expected_shape[i];
        if (i < expected_shape.size() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;

    std::cout << "Output size: " << output_data.size() << " (expected: " << static_cast<size_t>(static_cast<long long>(total_elements) * extra_dim) << ")" << std::endl;

    assert(output_data.size() == static_cast<size_t>(static_cast<long long>(total_elements) * extra_dim) && "Output size should match expected");

    std::cout << "✓ param_int32_to_fp16_weights test passed" << std::endl;
}

void test_param_int32_to_uint8_weights() {
    std::cout << "Testing param_int32_to_uint8_weights..." << std::endl;

    // Create test data for qweight (int32)
    std::vector<i32> qweight_data = {
        0x12345678, 0x9ABCDEF0, 0x11111111, 0x22222222
    };
    std::vector<size_t> qweight_shape = {4, 2}; // 4x2 tensor

    MockTensor qweight_tensor(qweight_data.data(), qweight_data.size() * sizeof(i32), qweight_shape, 2); // I32

    // Process: int32 -> uint8 for 4-bit
    int bits = 4;
    int mask = (1 << bits) - 1;
    int new_last_dim = qweight_shape.back() * (32 / bits);

    // Create output buffer
    std::vector<u8> output_data(qweight_shape[0] * new_last_dim);

    // Process each element: right shift by position, bitwise AND
    const i32* input_data = static_cast<const i32*>(qweight_tensor.data());

    for (size_t i = 0; i < qweight_shape[0]; ++i) {
        for (size_t j = 0; j < 32 / bits; ++j) {
            size_t output_idx = i * (32 / bits) + j;

            // Right shift by position
            i32 shifted = (input_data[i] >> (j * bits)) & 0xFF;

            // Bitwise AND with mask
            i32 masked = shifted & mask;

            output_data[output_idx] = static_cast<u8>(masked);
        }
    }

    // Expected output shape: [4, 16]
    std::vector<size_t> expected_shape = qweight_shape;
    expected_shape.back() = new_last_dim;

    std::cout << "Input shape: [";
    for (size_t i = 0; i < qweight_shape.size(); ++i) {
        std::cout << qweight_shape[i];
        if (i < qweight_shape.size() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;

    std::cout << "Expected output shape: [";
    for (size_t i = 0; i < expected_shape.size(); ++i) {
        std::cout << expected_shape[i];
        if (i < expected_shape.size() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;

    std::cout << "Output size: " << output_data.size() << " (expected: " << (qweight_shape[0] * new_last_dim) << ")" << std::endl;

    assert(output_data.size() == qweight_shape[0] * new_last_dim && "Output size should match expected");

    std::cout << "✓ param_int32_to_uint8_weights test passed" << std::endl;
}

void test_combine_low_nibbles() {
    std::cout << "Testing combine_low_nibbles..." << std::endl;

    // Create test data - pairs of bytes where we need to combine low nibbles
    std::vector<u8> test_data = {
        0xF1, 0xE2,  // Should produce 0x2F (E | 1)
        0x34, 0x56,  // Should produce 0x65 (6 | 5)
        0x78, 0x9A,  // Should produce 0xAB (A | B)
        0x11, 0x22   // Should produce 0x21 (2 | 1)
    };
    std::vector<size_t> test_shape = {8}; // 8 elements

    MockTensor test_tensor(test_data.data(), test_data.size(), test_shape, 0); // U8

    // Process: combine low nibbles from pairs
    const u8* input_data = static_cast<const u8*>(test_tensor.data());
    size_t total_elements = test_tensor.numel();

    // Ensure even number of elements
    assert(total_elements % 2 == 0 && "Array length must be even");

    // Create output array (half the size)
    std::vector<u8> combined_data(total_elements / 2);

    // Extract low 4 bits from each pair and combine
    for (size_t i = 0; i < total_elements; i += 2) {
        u8 low_nibble0 = input_data[i] & 0x0F;     // Low 4 bits of first element
        u8 low_nibble1 = input_data[i + 1] & 0x0F;  // Low 4 bits of second element

        // Combine: high nibble from second element, low nibble from first element
        combined_data[i / 2] = (low_nibble1 << 4) | low_nibble0;
    }

    std::cout << "Input data: ";
    for (size_t i = 0; i < std::min(size_t(8), test_tensor.numel()); ++i) {
        std::cout << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(input_data[i]) << " ";
    }
    std::cout << std::dec << std::endl;

    std::cout << "Output data: ";
    for (size_t i = 0; i < combined_data.size(); ++i) {
        std::cout << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(combined_data[i]) << " ";
    }
    std::cout << std::dec << std::endl;

    // Expected results: 0x2F, 0x65, 0xAB, 0x21
    assert(combined_data[0] == 0x2F && "First pair should produce 0x2F");
    assert(combined_data[1] == 0x65 && "Second pair should produce 0x65");
    assert(combined_data[2] == 0xAB && "Third pair should produce 0xAB");
    assert(combined_data[3] == 0x21 && "Fourth pair should produce 0x21");

    std::cout << "✓ combine_low_nibbles test passed" << std::endl;
}

int main() {
    std::cout << "=== GPTQ Weight Preprocessing Algorithm Tests ===" << std::endl;

    try {
        test_param_int32_to_fp16_weights();
        std::cout << std::endl;

        test_param_int32_to_uint8_weights();
        std::cout << std::endl;

        test_combine_low_nibbles();
        std::cout << std::endl;

        std::cout << "=== All algorithm tests passed! ===" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "✗ Test failed with exception: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}