import torch
from Encryption.Model1.XOR.LinearModel import LinearTimestep
from Encryption.Model1.XOR.Generation import XOR_image
from torchvision import transforms


def Generate_Sequences(key, seed):
    # Ensure key is a tensor
    key = torch.tensor([[key]], device='cuda')

    # Initialize the model and load weights
    model = LinearTimestep(1, 256 * 256).to('cuda')
    model.load_state_dict(torch.load(
        r"C:\Users\viraj\PycharmProjects\Research\Encryption\Model1\XOR\Models\model_-1.233387613818192e+18.pth"
    ))

    # Forward pass
    output = model(key)

    # Print output size
    print("Output Size:", output.size())

    # Calculate the min and max of the output
    output_min = output.min()
    output_max = output.max()
    print(output_min, output_max)
    # Normalize to range [0, 1] without clamping
    output_normalized = (output - output_min) / (output_max - output_min)
    output_scaled = output_normalized
    print(output_scaled[0][6000])
    # Calculate statistics
    mean = output_scaled.mean()  # Convert to float for mean calculation
    variance = output_scaled.var()
    std_dev = output_scaled.std()

    # Print a specific output value for reference
    print(f"Mean: {mean.item()}")
    print(f"Variance: {variance.item()}")
    print(f"Standard Deviation: {std_dev.item()}")
    return output_scaled.view(256, 256)


# Generate sequences with the given key and seed
x1 = Generate_Sequences(torch.randn(1, 1), 14345234)
x2 = Generate_Sequences(torch.randn(1, 1), 14345134)
