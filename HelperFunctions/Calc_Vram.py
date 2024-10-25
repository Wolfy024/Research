import torch
from Watermark.Model_1.BlendModels.MAM_encoder import MAM_encoder


def calculate_vram(model, input_shape, watermark_shape):
    # Example input shape: (batch_size, channels, height, width)
    batch_size, channels, height, width = input_shape
    batch_size_watermark, channels_watermark, height_watermark, width_watermark = watermark_shape

    # Calculate total parameters
    total_params = sum(p.numel() for p in model.parameters())

    # Calculate memory for parameters (float32, 4 bytes)
    param_memory = total_params * 4 / (1024 ** 2)  # Convert to MB

    # Create dummy inputs for both image and watermark
    dummy_image = torch.randn(input_shape).to(next(model.parameters()).device)  # Move to the same device as the model
    dummy_watermark = torch.randn(watermark_shape).to(next(model.parameters()).device)

    with torch.no_grad():
        _ = model(dummy_image, dummy_watermark)  # Forward pass with both inputs

    # Estimate memory for activations (float32, 4 bytes)
    # Note: The activations size can be more complicated to estimate accurately without keeping track of each layer's output.
    # Here we assume that activations are roughly similar in size to the parameters for a rough estimate.
    activations_memory = total_params * 4 / (1024 ** 2)  # Convert to MB

    # Total VRAM usage = parameters + activations
    total_memory = param_memory + activations_memory

    return {
        "Total Parameters (M)": total_params / 1e6,
        "Parameter Memory (MB)": param_memory,
        "Activations Memory (MB)": activations_memory,
        "Total Memory (MB)": total_memory
    }


# Example usage
model = MAM_encoder().to('cuda')  # Assuming you have a CUDA device
input_shape = (4, 3, 128, 128)  # Example input shape: 1 image, 3 channels, 128x128 size
watermark_shape = (4, 3, 128, 128)  # Example watermark shape: 1 watermark, 3 channels, 128x128 size
vram_usage = calculate_vram(model, input_shape, watermark_shape)
print(vram_usage)
