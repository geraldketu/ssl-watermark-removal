# Self-Supervised Watermark Removal from Graduation Photos

This project trains a deep learning model to remove proof watermarks from graduation photos without needing any real clean-watermarked image pairs. Instead, synthetic watermarks are applied to clean portrait images to create training data automatically.


---

## Background

Photography studios watermark graduation photos with text like PROOF, PREVIEW, or their studio name to prevent unpaid use. The problem with removing these watermarks using standard machine learning is that you need the same photo both with and without the watermark for training. Studios never release the clean originals, so building that kind of dataset is not possible.

This project gets around that by using self-supervised learning. Clean portrait images have synthetic watermarks added to them in code, and the model is trained to reverse that process. The watermarked image is the input and the original clean image is the label — all generated automatically with no manual labelling needed.
---

## How It Works

**1. Dataset creation**

Clean portrait images from [nielsr/CelebA-faces](https://huggingface.co/datasets/nielsr/CelebA-faces) are downloaded and resized to 256×256. A synthetic watermark generator written in Pillow applies randomised proof-style watermarks to each image, varying the text, layout, colour, opacity, rotation, and font size. JPEG compression noise is then applied to the watermarked image before saving. The result is 5,000 paired (clean, watermarked) images uploaded to Hugging Face as Parquet files.

**2. Model training**

A U-Net generator is trained to reconstruct the clean image from the watermarked input. The loss combines three terms:

- **L1 loss** — pixel-level reconstruction
- **Perceptual loss** — VGG16 feature comparison for sharper outputs
- **PatchGAN adversarial loss** — patch-level discriminator that pushes outputs to look more realistic

Training uses buffered data loading where 500 pairs are streamed into RAM at the start of each epoch to avoid reconnection overhead. Perceptual loss is computed every 4 steps to save around 30% of training time with no noticeable quality drop.

**3. Inference and post-processing**

The trained model runs on real graduation photos. A four-stage post-processing pipeline is applied on top of the model output:

1. **Watermark mask estimation** — subtracts a blurred version of the image from itself to locate where the watermark is
2. **Mask-guided blending** — uses the model output only where the mask detected a watermark, keeping original pixels everywhere else
3. **Unsharp masking** — sharpens fine texture the model softened during reconstruction
4. **Classical inpainting fallback** — OpenCV `INPAINT_TELEA` for photos where the neural model does not fully remove the watermark

---

## Setup

```bash
pip install torch torchvision torchmetrics pillow matplotlib \
            huggingface_hub datasets opencv-python-headless
```

All notebooks are designed to run on Google Colab with a GPU runtime. Mount your Google Drive and update the path variables at the top of each notebook to match your folder structure.

To load the trained model:

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


## Results

The model performs well on the synthetic test set, removing most watermark texture while keeping facial features and overall image composition intact. On real graduation photos the results are more limited because the real watermarks use different fonts and interact differently with complex backgrounds like graduation robes. The classical inpainting fallback performs better for high-contrast white proof text on plain backgrounds.

**Synthetic test set**

| Metric | Value    |
| ------ | -------- |
| PSNR   | 29.35 dB |
| SSIM   | 0.8893   |
| Val L1 | 0.0364   |

**Real photos** — evaluated qualitatively. The post-processing pipeline improves output quality noticeably, especially the mask-guided blending step which prevents the model from blurring areas of the image that do not have a watermark.

---

## What Changed Between v1 and v2

| Setting                 | v1              | v2                         |
| ----------------------- | --------------- | -------------------------- |
| Image resolution        | 128×128        | 256×256                   |
| Watermark opacity range | 180–250        | 80–220                    |
| JPEG compression noise  | No              | Yes                        |
| Colour watermarks       | No              | Yes (red, blue, gold)      |
| Validation split        | None            | 80/20                      |
| Loss function           | L1 + Perceptual | L1 + Perceptual + PatchGAN |
| Epochs                  | 30              | 50                         |
| Val PSNR                | ~34 dB          | 29.35 dB                   |
| Val SSIM                | ~0.96           | 0.8893                     |

The lower PSNR and SSIM in v2 is expected — the task is harder because the watermarks are more varied and the images include JPEG distortions that were not in v1.


---

## Known Issues

The PatchGAN discriminator loss flattened at around 0.71 early in training, which means the adversarial component stopped providing useful gradients. The reconstruction improvement came mainly from the L1 and perceptual losses. Future work would switch to a Wasserstein GAN objective, use label smoothing, and update the discriminator more frequently to keep the adversarial training stable.

---


## Repository Structure

```
ssl-watermark-removal/
├── v1/
│   ├── gerald_ndawula_dataset_Creation.ipynb
│   └── gerald_ndawula_unet_model.ipynb
│
└── v2/
    ├── 01_dataset_creation.ipynb
    ├── 02_model_training.ipynb
    ├── 03_evaluation.ipynb
    └── figure_generation.ipynb
```
## Links

| Resource      | Link                                                                                                   |
| ------------- | ------------------------------------------------------------------------------------------------------ |
| Dataset       | [GeraldNdawula/Watermark_Dataset_v2](https://huggingface.co/datasets/GeraldNdawula/Watermark_Dataset_v2)  |
| Model weights | [GeraldNdawula/Watermark_Removal_UNet_v2](https://huggingface.co/GeraldNdawula/Watermark_Removal_UNet_v2) |

---


## Citation

```
@misc{ndawula2026watermark,
  author = {Gerald Ndawula},
  title  = {Self-Supervised Watermark Removal from Graduation Photos},
  year   = {2026},
  url    = {https://github.com/geraldketu/ssl-watermark-removal}
}
```
