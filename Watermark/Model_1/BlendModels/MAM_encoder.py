from torch import nn
from torch import cat
import torch.nn.functional as F


class MAM_encoder(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MAM_encoder, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.Conv1 = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1)

    def forward(self, img, watermark):
        img_flat = img.view(img.size(0), -1, img.size(-1))
        watermark_flat = watermark.view(watermark.size(0), -1, watermark.size(-1))
        attn_output, _ = self.multihead_attn(img_flat, watermark_flat, watermark_flat)
        attn_output = attn_output.view(img.size(0), img.size(1), img.size(2), img.size(3))
        blend = self.layer_norm(img + attn_output)
        blend = self.Conv1(blend)
        return F.tanh(blend)

