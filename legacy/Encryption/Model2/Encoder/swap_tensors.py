import torch
from Encryption.Model2.Encoder.generate_seeds import generate_seed


def swap_tensors_row(image, sequences):
    image = image[0].clone()
    for i in range(image.size(0)):
        swap_idx = sequences[i]
        og_patch = image[i, :].clone()
        switch_patch = image[swap_idx, :].clone()
        image[i, :] = switch_patch
        image[swap_idx, :] = og_patch
    return image.unsqueeze(0)


def reverse_swap_tensors_row(image, sequences):
    image = image[0].clone()
    for reverse in range(image.size(0) - 1, -1, -1):
        swap_idx = sequences[reverse]
        og_patch = image[reverse, :].clone()
        switch_patch = image[swap_idx, :].clone()
        image[reverse, :] = switch_patch
        image[swap_idx, :] = og_patch
    return image.unsqueeze(0)


def swap_tensors_column(image, sequences):
    # Remove singleton dimension for processing
    image = image.squeeze()
    for i in range(image.size(1)):
        swap_idx = sequences[i]
        og_patch = image[:, i].clone()
        switch_patch = image[:, swap_idx].clone()
        image[:, i] = switch_patch
        image[:, swap_idx] = og_patch
    # Reshape to expected format
    image = image.unsqueeze(0)  # Add batch dimension back
    return image


def reverse_swap_tensors_column(image, sequences):
    # Remove singleton dimension for processing
    image = image.squeeze()
    for reverse in range(image.size(1) - 1, -1, -1):
        swap_idx = sequences[reverse]
        og_patch = image[:, reverse].clone()
        switch_patch = image[:, swap_idx].clone()
        image[:, reverse] = switch_patch
        image[:, swap_idx] = og_patch
    # Reshape to expected format
    image = image.unsqueeze(0)  # Add batch dimension back
    return image


def swap_tensors_individual(image, sequences):
    shape = image[0].shape
    image = image.view(-1).squeeze()
    for i in range(image.size(0)):
        swap_idx = sequences[i]
        og_patch = image[i].clone()
        switch_patch = image[swap_idx].clone()
        image[i] = switch_patch
        image[swap_idx] = og_patch
    image = image.view(shape).permute(1, 0, 2)
    return image


def reverse_swap_tensors_individual(image, sequences):
    shape = image[0].shape
    image = image.view(-1).squeeze()
    for reverse in range(image.size(0) - 1, -1, -1):
        swap_idx = sequences[reverse]
        og_patch = image[reverse].clone()
        switch_patch = image[swap_idx].clone()
        image[reverse] = switch_patch
        image[swap_idx] = og_patch
    image = image.view(shape).permute(1, 0, 2).unsqueeze(0)
    return image


def encrypt(bottleneck, x3, x2, x1):
    x1_shape = x1.shape
    x2_shape = x2.shape
    x3_shape = x3.shape
    bottleneck_shape = bottleneck.shape
    x1_encrypt = repeat_encrypt(x1, x1_shape, generate_seed())
    print('x1 done.')
    x2_encrypt = repeat_encrypt(x2, x2_shape, generate_seed())
    print('x2 done.')
    x3_encrypt = repeat_encrypt(x3, x3_shape, generate_seed())
    print('x3 done.')
    bottleneck_encrypt = repeat_encrypt(bottleneck, bottleneck_shape, generate_seed())
    print('bottleneck done.')
    return bottleneck_encrypt, x3_encrypt, x2_encrypt, x1_encrypt


def decrypt(bottleneck, x3, x2, x1, keys: list):
    x1_shape = x1.shape
    x2_shape = x2.shape
    x3_shape = x3.shape
    bottleneck_shape = bottleneck.shape
    x1_encrypt = repeat_decrypt(x1, x1_shape, keys[0])
    print('x1 done.')
    x2_encrypt = repeat_decrypt(x2, x2_shape, keys[1])
    print('x2 done.')
    x3_encrypt = repeat_decrypt(x3, x3_shape, keys[2])
    print('x3 done.')
    bottleneck_encrypt = repeat_decrypt(bottleneck, bottleneck_shape, keys[3])
    print('bottleneck done.')
    return bottleneck_encrypt, x3_encrypt, x2_encrypt, x1_encrypt


def repeat_decrypt(image, shape, seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    seq1 = torch.randperm(shape[1])
    seq2 = torch.randperm(shape[2] * shape[3])
    seq3 = torch.randperm(shape[1] * shape[2] * shape[3])
    image = image.view(1, shape[1], 1, shape[2] * shape[3])
    image = reverse_swap_tensors_individual(image, seq3)
    image = image.view(1, shape[1], 1, shape[2] * shape[3])
    image = reverse_swap_tensors_column(image, seq2)
    image = image.permute(1, 0, 2).unsqueeze(0)
    image = swap_tensors_row(image, seq1)
    image = image.view(shape)
    return image


def repeat_encrypt(image, shape, seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    seq1 = torch.randperm(shape[1])
    seq2 = torch.randperm(shape[2] * shape[3])
    seq3 = torch.randperm(shape[1] * shape[2] * shape[3])
    image = image.view(1, shape[1], 1, shape[2] * shape[3])
    image = reverse_swap_tensors_individual(image, seq3)
    image = reverse_swap_tensors_column(image, seq2)
    image = image.permute(1, 0, 2).unsqueeze(0)
    image = reverse_swap_tensors_row(image, seq1)
    image = image.view(shape)
    return image


def test_enc(image, shape, seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    seq_1 = torch.randperm(shape[1])
    seq_2 = torch.randperm(shape[2] * shape[3])
    seq_3 = torch.randperm(shape[1] * shape[2] * shape[3])
    image = image.view(1, shape[1], 1, shape[2] * shape[3])
    image = swap_tensors_row(image, seq_1)
    image = swap_tensors_column(image, seq_2)
    image = image.permute(1, 0, 2).unsqueeze(0)
    image = swap_tensors_individual(image, seq_3)
    image = image.permute(1, 0, 2).unsqueeze(0)
    image = reverse_swap_tensors_individual(image, seq_3)
    image = reverse_swap_tensors_column(image, seq_2)
    image = image.permute(1, 0, 2).unsqueeze(0)
    image = reverse_swap_tensors_row(image, seq_1)
    are_equal = torch.equal(OG_image, image)
    print(are_equal)


if __name__ == "__main__":
    a = torch.randint(0, 10, (1, 4, 5, 5))
    test_enc(a, a.shape, generate_seed())
    # b = torch.randint(0, 10, (1, 4, 5, 5))
    # a = torch.randint(0, 10, (1, 4, 5, 5))
    # seq1 = torch.tensor([2, 3, 1, 0])
    # seq2 = torch.randperm(25)
    # seq3 = torch.randperm(100)
    # print(f'Original Tensor: {a}')  ##################
    # a = a.view(1, 4, 1, 25)
    # a = swap_tensors_row(a, seq1)
    # a = a.view(1, 4, 1, 25)
    # a = swap_tensors_column(a, seq2)
    # a = a.permute(1, 0, 2).unsqueeze(0)
    # a = swap_tensors_individual(a, seq3)
    # print(f'swapped Tensor: {a.view(1, 4, 5, 5)}')
    # a = a.permute(1, 0, 2).unsqueeze(0)
    # a = reverse_swap_tensors_individual(a, seq3)
    # a = reverse_swap_tensors_column(a, seq2)
    # a = reverse_swap_tensors_row(a, seq1)
    # print(f'Original Tensor: {a.view(1, 4, 5, 5)}')  ##################
