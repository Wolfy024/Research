import torch


def xor_image_with_tensor(image_tensor, xor_tensor):
    """
    XORs an image tensor with a given tensor.

    Args:
        image_tensor (torch.Tensor): The input image tensor.
        xor_tensor (torch.Tensor): The tensor to XOR with the image.

    Returns:
        torch.Tensor: The XORed result.
    """
    if image_tensor.shape != xor_tensor.shape:
        raise ValueError("The dimensions of image_tensor and xor_tensor must be the same.")

    # Perform the XOR operation
    xor_result = image_tensor ^ xor_tensor  # Using bitwise XOR

    return xor_result
