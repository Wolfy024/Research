import torch

def generate_sequences(seeds: list, max_length: int):
    sequences = torch.empty((512, 32, 32))
    for i in range(len(seeds)):
        for j in range(max_length):
            torch.manual_seed(seeds[i][j])
            sequence = torch.randperm(max_length)
            sequences[i, j] = sequence
    return sequences
