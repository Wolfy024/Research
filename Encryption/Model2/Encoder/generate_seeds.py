import torch
import secrets


def generate_seed():
    hex_key = secrets.token_hex(16)  # Generate a 16-byte hex key
    seed = int(hex_key, 16) % (2 ** 32)  # Convert to an integer and reduce to 32 bits
    print(seed)
    return seed