import torch


def xor_image_tensor(image_tensor, xor_key):
    """
    Applies a pixel-wise XOR operation between an image tensor and a key tensor.

    Args:
        image_tensor (torch.Tensor): The input image tensor with shape [C, H, W] and dtype int or uint8, values in 0–255.
        xor_key (torch.Tensor): The XOR key tensor with shape [C, H, W], should match image tensor in shape and dtype.

    Returns:
        torch.Tensor: The XORed image tensor with the same shape as image_tensor.
    """
    # Ensure both tensors are on the same device
    xor_key = xor_key.to(image_tensor.device)

    # Apply element-wise XOR between image and key
    xor_result = torch.bitwise_xor(image_tensor, xor_key)

    return xor_result
