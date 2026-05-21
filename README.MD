# Self-Supervised Watermark Removal from Graduation Photos

> Remove proof watermarks from graduation photos using a self-supervised deep learning pipeline — no real clean-watermarked pairs needed.

<p align="left">
  <a href="https://huggingface.co/datasets/GeraldNdawula/Watermark_Dataset_v2">
    <img src="https://img.shields.io/badge/🤗 Dataset-Watermark_Dataset_v2-blue" />
  </a>
  <a href="https://huggingface.co/GeraldNdawula/Watermark_Removal_UNet_v2">
    <img src="https://img.shields.io/badge/🤗 Model-Watermark_Removal_UNet_v2-orange" />
  </a>
  <img src="https://img.shields.io/badge/Framework-PyTorch-red" />
  <img src="https://img.shields.io/badge/Runtime-Google Colab A100-yellow" />
</p>

---

## Overview

Photography studios watermark graduation photos with text like **PROOF**, **PREVIEW**, or their studio name to prevent unpaid use. Removing these watermarks with standard supervised learning requires the same photo both with and without the watermark — but studios never release the clean originals.

This project solves that with **self-supervised learning**. Synthetic watermarks are applied to clean portrait images in code, and the model is trained to reverse that process. No manual labelling needed.

---

## How It Works

### 1 · Dataset Creation

Clean portrait images from [nielsr/CelebA-faces](https://huggingface.co/datasets/nielsr/CelebA-faces) are resized to 256×256. A synthetic watermark generator written in Pillow randomises the text, layout, colour, opacity, rotation, and font size for every image. JPEG compression noise is applied before saving to better match real studio photos. The result is **5,000 paired (clean, watermarked) images** uploaded to Hugging Face as Parquet files.

### 2 · Model Training

A **U-Net generator** is trained to reconstruct the clean image from the watermarked input using three combined loss terms:

| Loss                 | Purpose                                        |
| -------------------- | ---------------------------------------------- |
| L1                   | Pixel-level reconstruction — fast convergence |
| Perceptual (VGG16)   | Feature-level comparison — sharper outputs    |
| PatchGAN adversarial | Patch-level realism — reduces blurring        |

Training uses buffered data loading (500 pairs per epoch streamed into RAM) to avoid reconnection overhead. Perceptual loss runs every 4 steps, saving ~30% compute with no quality drop.

### 3 · Inference + Post-Processing

A four-stage pipeline is applied on top of the model output to improve real-world results:

```
Watermarked photo
       │
       ▼
  [1] Watermark mask estimation   — Gaussian blur subtraction to find watermark pixels
       │
       ▼
  [2] Mask-guided blending        — model output used only where mask is active
       │
       ▼
  [3] Unsharp masking             — recovers fine texture the model softened
       │
       ▼
  [4] Classical inpainting        — OpenCV INPAINT_TELEA fallback for stubborn text
       │
       ▼
  Cleaned output
```

---

## Repository Structure

```
ssl-watermark-removal/
├── v1/                                          ← Proof of concept (128×128, L1 + Perceptual)
│   ├── gerald_ndawula_dataset_Creation.ipynb
│   └── gerald_ndawula_unet_model.ipynb
│
└── v2/                                          ← Improved pipeline (256×256, + PatchGAN)
    ├── 01_dataset_creation.ipynb
    ├── 02_model_training.ipynb
    ├── 03_evaluation.ipynb
    └── figure_generation.ipynb
```

---

## v1 vs v2

| Setting           | v1              | v2                                   |
| ----------------- | --------------- | ------------------------------------ |
| Resolution        | 128×128        | **256×256**                   |
| Opacity range     | 180–250        | **80–220**                    |
| JPEG noise        | ✗              | **✓**                         |
| Colour watermarks | ✗              | **✓** (red, blue, gold)       |
| Validation split  | ✗              | **80 / 20**                    |
| Loss              | L1 + Perceptual | **L1 + Perceptual + PatchGAN** |
| Epochs            | 30              | **50**                         |
| Val PSNR          | ~34 dB          | 29.35 dB                             |
| Val SSIM          | ~0.96           | 0.8893                               |

> The lower PSNR and SSIM in v2 is expected — the task is harder because the watermarks are more varied and include JPEG distortions not present in v1.

---

## Results

### Synthetic test set

| Metric | v2       |
| ------ | -------- |
| PSNR   | 29.35 dB |
| SSIM   | 0.8893   |
| Val L1 | 0.0364   |

### Real graduation photos

Evaluated qualitatively. The model partially removes watermarks on real photos but does not fully eliminate high-contrast proof text in all cases due to the domain gap between synthetic training data and real studio images. The classical inpainting fallback handles these cases well. The mask-guided blending step prevents the model from blurring clean areas of the image.

---

## Setup

```bash
pip install torch torchvision torchmetrics pillow matplotlib \
            huggingface_hub datasets opencv-python-headless
```

All notebooks run on **Google Colab** with a GPU runtime. Mount your Drive and update the path variables at the top of each notebook.

### Load the trained model

```python
from huggingface_hub import hf_hub_download
import torch

model_path = hf_hub_download(
    repo_id="GeraldNdawula/Watermark_Removal_UNet_v2",
    filename="unet_v2_final.pth"
)

ckpt  = torch.load(model_path, map_location="cpu")
model = UNet()
model.load_state_dict(ckpt["generator"])
model.eval()
```

---

## Known Issues

The PatchGAN discriminator loss flattened at ~0.71 early in training, meaning the adversarial component stopped providing useful gradients. Reconstruction improvement came mainly from L1 and perceptual losses.

Planned fixes:

- Switch to **Wasserstein GAN** with gradient penalty for more stable training
- Apply **label smoothing** to keep the discriminator from becoming overconfident
- Update the discriminator **more frequently** than the generator rather than just using a higher learning rate
- Add **spectral normalisation** to constrain discriminator weight updates

---

## Citation

```bibtex
@misc{ndawula2026watermark,
  author = {Gerald Ndawula},
  title  = {Self-Supervised Watermark Removal from Graduation Photos},
  year   = {2026},
  url    = {https://github.com/geraldketu/ssl-watermark-removal}
}
```
