"""
Example demonstrating weight preprocessing in InfiniLM.

This example shows how to use the weight preprocessing functionality
similar to vLLM's process_weights_after_loading.
"""

import torch
from infinilm import InferEngine, AutoConfig, DistConfig
import infinicore


def load_pytorch_checkpoint(model_path):
    """
    Load a model checkpoint from PyTorch format.

    Args:
        model_path: Path to the model checkpoint

    Returns:
        Dictionary mapping parameter names to tensors
    """
    # Load the checkpoint
    checkpoint = torch.load(model_path, map_location="cpu")

    # Convert to InfiniLM tensors
    state_dict = {}
    for name, param in checkpoint.items():
        # Convert PyTorch tensor to numpy, then to InfiniLM tensor
        numpy_param = param.cpu().numpy()
        infini_tensor = infinicore.from_numpy(numpy_param)
        state_dict[name] = infini_tensor

    return state_dict


def example_basic_usage():
    """Basic example of weight preprocessing."""
    print("Example 1: Basic Usage")

    # Create engine
    model_path = "path/to/your/model"
    config = AutoConfig.from_pretrained(model_path)

    engine = InferEngine(
        model_path=model_path,
        device=infinicore.device(),
        distributed_config=DistConfig(1),  # Single GPU
    )

    # Load weights
    checkpoint_path = "path/to/checkpoint.bin"
    print(f"Loading checkpoint from {checkpoint_path}...")
    state_dict = load_pytorch_checkpoint(checkpoint_path)

    print("Loading weights into engine...")
    engine.load_state_dict(state_dict)

    # Preprocess weights - this is the key step!
    print("Preprocessing weights...")
    success = engine.preprocess_weights()

    if success:
        print("✓ Weight preprocessing completed successfully")
    else:
        print("✗ Weight preprocessing had some failures")

    # Now the model is ready for inference
    print("Model is ready for inference!")


def example_with_quantization():
    """Example with GPTQ quantization."""
    print("\nExample 2: GPTQ Quantization")

    # For quantized models
    model_path = "path/to/gptq/model"
    config = AutoConfig.from_pretrained(model_path)

    # Check quantization scheme
    if hasattr(config, 'quantization'):
        print(f"Detected quantization scheme: {config.quantization}")

    engine = InferEngine(
        model_path=model_path,
        device=infinicore.device(),
    )

    # Load GPTQ checkpoint
    print("Loading GPTQ checkpoint...")
    state_dict = load_pytorch_checkpoint("path/to/gptq_checkpoint.bin")

    print("Loading weights into engine...")
    engine.load_state_dict(state_dict)

    # Preprocess weights - GPTQ-specific optimizations will be applied
    print("Preprocessing GPTQ weights...")
    print("This will optimize:")
    print("  - Packed qweight layout")
    print("  - Scale and zero point storage")
    print("  - Activation ordering (g_idx) if present")

    success = engine.preprocess_weights()

    if success:
        print("✓ GPTQ weight preprocessing completed successfully")
    else:
        print("✗ GPTQ weight preprocessing had some failures")

    print("GPTQ model is ready for inference!")


def example_multi_gpu():
    """Example with multi-GPU (tensor parallelism)."""
    print("\nExample 3: Multi-GPU with Tensor Parallelism")

    # Create engine with 4 GPUs
    model_path = "path/to/model"
    engine = InferEngine(
        model_path=model_path,
        device=infinicore.device(),
        distributed_config=DistConfig(tp_device_ids=[0, 1, 2, 3]),
    )

    # Load weights
    state_dict = load_pytorch_checkpoint("path/to/checkpoint.bin")
    engine.load_state_dict(state_dict)

    # Preprocess weights - will be applied on each GPU independently
    print("Preprocessing weights on 4 GPUs...")
    success = engine.preprocess_weights()

    if success:
        print("✓ Weight preprocessing completed successfully on all GPUs")
    else:
        print("✗ Weight preprocessing had some failures on one or more GPUs")

    print("Multi-GPU model is ready for inference!")


def example_with_error_handling():
    """Example with comprehensive error handling."""
    print("\nExample 4: With Error Handling")

    try:
        # Create engine
        model_path = "path/to/model"
        engine = InferEngine(
            model_path=model_path,
            device=infinicore.device(),
        )

        # Load weights
        state_dict = load_pytorch_checkpoint("path/to/checkpoint.bin")
        engine.load_state_dict(state_dict)

        # Preprocess weights with error handling
        print("Preprocessing weights...")
        success = engine.preprocess_weights()

        if not success:
            print("Warning: Weight preprocessing had some failures")
            print("Model may still work, but performance could be degraded")
            print("Check logs for details")

        # Verify preprocessing by checking parameter names
        param_names = engine.state_dict_keyname()
        print(f"Model has {len(param_names)} parameters")

        # Test inference with a simple input
        print("Testing inference...")
        input_ids = infinicore.from_list([[1, 2, 3]], dtype=infinicore.int64)

        output = engine.forward(
            input_ids=input_ids,
            position_ids=infinicore.from_list([[0, 1, 2]], dtype=infinicore.int64),
        )

        print("✓ Inference test passed!")

    except Exception as e:
        print(f"✗ Error occurred: {e}")
        print("Please check:")
        print("  1. Model path is correct")
        print("  2. Checkpoint format is compatible")
        print("  3. Quantization scheme is properly configured")
        raise


if __name__ == "__main__":
    print("=" * 60)
    print("InfiniLM Weight Preprocessing Examples")
    print("=" * 60)

    # Run examples (comment out the ones you don't want to run)
    # example_basic_usage()
    # example_with_quantization()
    # example_multi_gpu()
    # example_with_error_handling()

    print("\n" + "=" * 60)
    print("Examples completed!")
    print("=" * 60)
    print("\nNote: Update the paths in each example to match your setup")