import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import sys
import os
import argparse

# ── Architecture must match training exactly ──────────────────────────────────

class DenseBlock(nn.Module):
    def __init__(self, in_channels=64, growth_rate=32):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, growth_rate, 3, padding=1)
        self.conv2 = nn.Conv2d(in_channels + growth_rate * 1, growth_rate, 3, padding=1)
        self.conv3 = nn.Conv2d(in_channels + growth_rate * 2, growth_rate, 3, padding=1)
        self.conv4 = nn.Conv2d(in_channels + growth_rate * 3, growth_rate, 3, padding=1)
        self.conv5 = nn.Conv2d(in_channels + growth_rate * 4, in_channels, 3, padding=1)
        self.lrelu = nn.LeakyReLU(0.2, inplace=False)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat([x, x1], 1)))
        x3 = self.lrelu(self.conv3(torch.cat([x, x1, x2], 1)))
        x4 = self.lrelu(self.conv4(torch.cat([x, x1, x2, x3], 1)))
        x5 = self.conv5(torch.cat([x, x1, x2, x3, x4], 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    def __init__(self, in_channels=64, growth_rate=32):
        super().__init__()
        self.db1 = DenseBlock(in_channels, growth_rate)
        self.db2 = DenseBlock(in_channels, growth_rate)
        self.db3 = DenseBlock(in_channels, growth_rate)

    def forward(self, x):
        out = self.db1(x)
        out = self.db2(out)
        out = self.db3(out)
        return out * 0.2 + x


class Generator(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, num_features=64, num_rrdb=16):
        super().__init__()
        self.conv_first      = nn.Conv2d(in_channels, num_features, 3, padding=1)
        self.trunk           = nn.Sequential(*[RRDB(num_features, 32) for _ in range(num_rrdb)])
        self.conv_last_trunk = nn.Conv2d(num_features, num_features, 3, padding=1)
        self.up1_conv        = nn.Conv2d(num_features, num_features * 4, 3, padding=1)
        self.up1_ps          = nn.PixelShuffle(2)
        self.up2_conv        = nn.Conv2d(num_features, num_features * 4, 3, padding=1)
        self.up2_ps          = nn.PixelShuffle(2)
        self.hr_conv         = nn.Conv2d(num_features, num_features, 3, padding=1)
        self.conv_last       = nn.Conv2d(num_features, out_channels, 3, padding=1)
        self.skip_hr_conv    = nn.Conv2d(num_features, num_features, 3, padding=1)
        self.skip_conv_last  = nn.Conv2d(num_features, out_channels, 3, padding=1)
        self.lrelu           = nn.LeakyReLU(0.2)
        self.tanh            = nn.Tanh()

    def forward(self, x):
        feat_first = self.conv_first(x)
        trunk_out  = self.conv_last_trunk(self.trunk(feat_first))
        feat       = feat_first + trunk_out

        feat = F.leaky_relu(self.up1_ps(self.up1_conv(feat)), 0.2, inplace=False)
        feat = F.leaky_relu(self.up2_ps(self.up2_conv(feat)), 0.2, inplace=False)
        feat = F.leaky_relu(self.hr_conv(feat),               0.2, inplace=False)
        out  = self.tanh(self.conv_last(feat))

        # Skip connection — must match training
        skip = F.interpolate(feat_first, scale_factor=4, mode="bilinear", align_corners=False)
        skip = F.leaky_relu(self.skip_hr_conv(skip), 0.2, inplace=False)
        skip = self.tanh(self.skip_conv_last(skip))

        return torch.clamp(out + skip, -1.0, 1.0)


# ── Config ────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="4x plant leaf super-resolution (32x32 -> 128x128)")
parser.add_argument("--weights", default="checkpoints/best_generator.pth")
parser.add_argument("--image",   default="results/agrivision_test_0013.png")
parser.add_argument("--out",     default="results/comparison.png")
args = parser.parse_args()

WEIGHTS_PATH = args.weights
IMG_PATH     = args.image

# ── Load model ────────────────────────────────────────────────────────────────

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

if not os.path.exists(WEIGHTS_PATH):
    print(f"[ERROR] Weights not found: {WEIGHTS_PATH}")
    sys.exit(1)

generator = Generator(num_rrdb=16).to(device)
generator.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
generator.eval()
print("[OK] Generator loaded")

# ── Load & preprocess image ───────────────────────────────────────────────────

if not os.path.exists(IMG_PATH):
    print(f"[ERROR] Image not found: {IMG_PATH}")
    sys.exit(1)

img       = Image.open(IMG_PATH).convert("RGB")
lr_32     = img.resize((32, 32), Image.BILINEAR)   # ensure 32×32 input
print(f"[OK] Image loaded: original {img.size} -> resized to 32×32")

lr_tensor = (
    torch.tensor(np.array(lr_32, dtype=np.float32) / 127.5 - 1.0)
    .permute(2, 0, 1)
    .unsqueeze(0)
    .to(device)
)

# ── Inference ─────────────────────────────────────────────────────────────────

with torch.no_grad():
    sr_tensor = generator(lr_tensor)

sr_np = (
    (sr_tensor.squeeze().permute(1, 2, 0).cpu().numpy() + 1.0) * 127.5
).clip(0, 255).astype(np.uint8)

print(f"[OK] SR output: shape={sr_np.shape}, range=[{sr_np.min()}, {sr_np.max()}]")

# ── Plot: LR | Bicubic | ESRGAN ───────────────────────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(12, 4))

axes[0].imshow(lr_32)
axes[0].set_title("Input LR (32×32)", fontsize=12)

axes[1].imshow(lr_32.resize((128, 128), Image.NEAREST))
axes[1].set_title("Bicubic Upscale (128×128)", fontsize=12)

axes[2].imshow(sr_np)
axes[2].set_title("ESRGAN Output (128×128)", fontsize=12)

for ax in axes:
    ax.axis("off")

plt.suptitle("Plant Leaf Super Resolution", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(args.out, dpi=150, bbox_inches="tight")
print(f"[SAVED] {args.out}")
plt.show()