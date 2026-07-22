from Encryption.Model2.Compile import bottleneck, encode_image, Reverse
import torch
from Encryption.Model2.Encoder import generate_sequences, generate_seeds, swap_tensors
from PIL import Image
from torchvision import transforms
from cryptography.fernet import Fernet
import io
def encrypt(image_path):
    neck, x3, x2, x1 = bottleneck.get_bottleneck(image_path)
    encoded_image = swap_tensors.encrypt(neck, x3, x2, x1)
    key = "oODE8oyR8TsEmICEoGMDZoc8g--d3m9-q0bxEusKrJM="
    key = Fernet.generate_key()
    save_encrypted_model(encoded_image, "encrypted_tuple.pt", key)
    load_encrypted_model("encrypted_tuple.pt", key)


def save_encrypted_model(model, file_path, key):
    cipher_suite = Fernet(key)
    buffer = io.BytesIO()
    torch.save(model, buffer)
    buffer.seek(0)
    model_data = buffer.read()
    encrypted_data = cipher_suite.encrypt(model_data)
    with open(file_path, "wb") as f:
        f.write(encrypted_data)


def load_encrypted_model(file_path, key):
    cipher_suite = Fernet(key)
    with open(file_path, "rb") as f:
        encrypted_data = f.read()
    decrypted_data = cipher_suite.decrypt(encrypted_data)
    buffer = io.BytesIO(decrypted_data)
    model = torch.load(buffer)
    torch.save(model, "decrypted_model.pt")

encrypt(r"C:\Users\viraj\PycharmProjects\Research\data\Abyssinian_20.jpg")

