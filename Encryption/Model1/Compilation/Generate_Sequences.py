from Encryption.Model1.Encoder.Encoder import Encoder
import torch
from Encryption.Model1.ChaoticSequence.ChaosModel import ChaosLSTM
from Encryption.Model1.XorSequence.xor_model import XORTrainer


def generate_keys(key: float, seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    model_Encoder = Encoder(1, 3, 256, 256).to('cuda')
    model_path_encoder = r'C:\Users\viraj\PycharmProjects\Research\Encryption\Model1\Encoder\Models\encoder_5_-453522895.25.pt'
    model_Encoder.load_state_dict(torch.load(model_path_encoder))
    time_steps_channel1 = model_Encoder(torch.tensor(float(key),
                                                     dtype=torch.float32).unsqueeze(0).to('cuda')).to('cuda')
    del model_Encoder
    torch.cuda.empty_cache()
    time_steps_channels = time_steps_channel1.view(3, 256, 256)
    time_steps_channel1 = time_steps_channels[0]
    time_steps_channel2 = time_steps_channels[1]
    time_steps_channel3 = time_steps_channels[2]
    return time_steps_channel1, time_steps_channel2, time_steps_channel3


def generate_chaotic_sequences(input_1, seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    model_Chaos = ChaosLSTM(1, 1024, 3).to('cuda')
    model_path_chaos = r'C:\Users\viraj\PycharmProjects\Research\Encryption\Model1\ChaoticSequence\Models\model_11_-18.13003482669592.pt'
    model_Chaos.load_state_dict(torch.load(model_path_chaos))
    user_input_key_1 = torch.tensor(float(input_1), dtype=torch.float32).unsqueeze(0).to('cuda')
    user_input_key_1 = model_Chaos(user_input_key_1).to('cuda')
    return user_input_key_1


def generate_xor_keys(key: float, seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    model_Encoder = XORTrainer(1, 3, 256).to('cuda')
    model_path_encoder = r'C:\Users\viraj\PycharmProjects\Research\Encryption\Model1\XorSequence\Models\encoder_1_-13545578.646972656.pt'
    model_Encoder.load_state_dict(torch.load(model_path_encoder))
    time_steps_channel1 = model_Encoder(torch.tensor(float(key),
                                                     dtype=torch.float32).unsqueeze(0).to('cuda')).to('cuda')
    del model_Encoder
    torch.cuda.empty_cache()
    time_steps_channels = time_steps_channel1.view(3, 256, 256)
    return time_steps_channels


def generate_confusion_keys(X, seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # Generate a tensor with values from 1 to X
    sequence = torch.arange(0, X + 1)
    # Shuffle the sequence based on the seed
    shuffled_sequence = sequence[torch.randperm(X)]
    return shuffled_sequence
