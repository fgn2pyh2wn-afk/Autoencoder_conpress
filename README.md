# AE-Compress

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-ee4c2c.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)]()

A compact **convolutional autoencoder** in pure PyTorch that learns to compress and reconstruct [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist) images.

Each `28 × 28` grayscale image is squeezed into a **20-dimensional latent vector** (a **~39× reduction** at the feature level), then rebuilt by a transposed-convolution decoder. The entire pipeline — data loading, training, evaluation, and visualization — lives in a single notebook: `AE_conpress.ipynb`.

## Highlights

- ⚙️ Pure PyTorch — only `torch`, `torchvision`, `matplotlib` required
- 🍎 Automatic device selection: Apple Silicon **MPS** acceleration, CPU fallback
- 🧠 Convolutional encoder + transposed-convolutional decoder with a 20-dim latent bottleneck
- 📉 Training loss drops from **0.3308 → 0.2698** in 20 epochs (BCELoss, Adam)
- 🖼️ Built-in side-by-side visualization of original vs. reconstructed images

## Model Architecture

```
 Input 28×28×1
        │
        ▼
 ┌──────────────── Encoder ────────────────┐
 │ Conv2d(1 → 32, k=3, s=2, p=1) + ReLU    │  → 14×14×32
 │ Conv2d(32 → 64, k=3, s=2, p=1) + ReLU   │  → 7×7×64
 │ Flatten(64×7×7) → Linear → latent(20)   │  bottleneck
 └─────────────────────────────────────────┘
        │
        ▼
 ┌──────────────── Decoder ────────────────┐
 │ Linear(20) → 64×7×7                     │
 │ ConvTranspose2d(64 → 32, k=3, s=2) + ReLU        │  → 14×14×32
 │ ConvTranspose2d(32 → 1, k=3, s=2) + Sigmoid      │  → 28×28×1
 └─────────────────────────────────────────┘
        │
        ▼
 Output 28×28×1
```

| Component | Detail |
|---|---|
| Encoder | 2 × stride-2 Conv + ReLU, then FC → `latent_dim = 20` |
| Bottleneck | 20-dim latent vector (`784 → 20`, ~39× compression) |
| Decoder | FC → `64×7×7`, 2 × ConvTranspose + ReLU, Sigmoid output |
| Loss | `BCELoss` |
| Optimizer | `Adam`, lr = `1e-3` |
| Batch size | `128` |
| Epochs | `20` |

## Getting Started

### Requirements

- Python 3.9+
- PyTorch (MPS support recommended on Apple Silicon)
- torchvision
- matplotlib

```bash
pip install torch torchvision matplotlib
```

### Run

```bash
jupyter notebook AE_conpress.ipynb
```

Run all cells. The notebook will:

1. Automatically download Fashion-MNIST (cached in `./data`)
2. Train the autoencoder for 20 epochs, printing the average loss per epoch
3. Reconstruct the first test batch and show 8 original / reconstructed pairs

Alternatively, convert to a plain script:

```bash
jupyter nbconvert --to script AE_conpress.ipynb
```

## Training Results

Training on **MPS** (Apple Silicon), 20 epochs, `BCELoss`:

| Epoch | Loss | Epoch | Loss |
|------:|-----:|------:|-----:|
| 1  | 0.3308 | 11 | 0.2719 |
| 2  | 0.2850 | 12 | 0.2715 |
| 3  | 0.2801 | 13 | 0.2712 |
| 4  | 0.2777 | 14 | 0.2710 |
| 5  | 0.2761 | 15 | 0.2707 |
| 6  | 0.2749 | 16 | 0.2705 |
| 7  | 0.2740 | 17 | 0.2703 |
| 8  | 0.2733 | 18 | 0.2701 |
| 9  | 0.2727 | 19 | 0.2699 |
| 10 | 0.2723 | 20 | 0.2698 |

![Training loss](training_loss.png)

Loss drops sharply in the first few epochs, then converges smoothly — no sign of overfitting or divergence.

<img width="1350" height="750" alt="image" src="https://github.com/user-attachments/assets/29bcfb21-61e3-47d9-8f57-a3a01c72e2e8" />

## Results

After 20 epochs the decoder produces clearly recognizable reconstructions: item silhouettes and texture structure are preserved despite the aggressive 20-dim bottleneck, demonstrating that a low-dimensional latent space is sufficient to capture the dominant visual variation of Fashion-MNIST.

The notebook's final cell renders an 8-sample comparison (originals on the top row, reconstructions on the bottom row).

## Project Structure

```
.
├── AE_conpress.ipynb   # Full pipeline: training + evaluation + visualization
├── training_loss.png   # Loss curve (generated from the logged training output)
└── README.md
```

## License

[MIT](LICENSE)
