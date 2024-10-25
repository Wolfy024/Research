import torch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LEARNING_RATE = 2e-4
BATCH_SIZE = 16
NUM_EPOCHS = 20
TRAIN_RATIO = 0.8