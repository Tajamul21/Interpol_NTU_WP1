# 🌟 NTU Interop WP1 — Qwen3-VL Multimodal Pipeline

> ⚡ High-performance multi-GPU vision-language pipeline for detection, reasoning, and OCR using Qwen3-VL

---

## 🧠 Overview

Multimodal pipeline using **Qwen3-VL-8B-Instruct** for:
- Detection
- OCR
- Visual reasoning
- Multi-GPU scaling

---

## ⚙️ Installation (Qwen3-VL Setup)

### 1. Create Environment
\`\`\`bash
conda create -n qwen3vl_py311 python=3.11 -y
conda activate qwen3vl_py311
\`\`\`

---

### 2. Install PyTorch (IMPORTANT)

👉 Match your CUDA version (example below = CUDA 12.1)

\`\`\`bash
pip install torch==2.6.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
\`\`\`

> Qwen3 requires **torch >= 2.6** :contentReference[oaicite:0]{index=0}

---

### 3. Install Transformers + Core Dependencies

\`\`\`bash
pip install transformers>=4.51.0 accelerate sentencepiece
pip install pillow tqdm
\`\`\`

> Qwen3-VL requires **new Transformers (≥4.5x)** for model support :contentReference[oaicite:1]{index=1}

---

### 4. Install Optional Speedups (Recommended)

#### Flash Attention 2 (Huge speed boost)
\`\`\`bash
pip install flash-attn --no-build-isolation
\`\`\`

- 2–4× faster attention
- lower memory usage :contentReference[oaicite:2]{index=2}

---

### 5. Hugging Face Login (Recommended)

\`\`\`bash
huggingface-cli login
\`\`\`

---

### 6. Verify Setup

\`\`\`python
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
import torch

model = Qwen3VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen3-VL-8B-Instruct",
    torch_dtype=torch.float16,
    device_map="auto"
)
print("Model loaded successfully")
\`\`\`

---

## 🚀 Running

### Single GPU
\`\`\`bash
python run.py
\`\`\`

---

### Multi-GPU
\`\`\`bash
python run_parallel.py \\
  --gpu-ids 0,4,6,7 \\
  --batch-size 4 \\
  --allow-tf32
\`\`\`

---

## ⚡ Key Notes

- Each GPU loads its **own model copy**
- Each image → **3 model calls**
- Batch size = **queue batching (NOT real batching)**
- Flash Attention = **major speed improvement**

---

## 🎯 Hardware

- GPU required (recommended ≥16GB VRAM)
- CUDA ≥ 11.0
- Flash-attn requires modern GPU (Ampere+) :contentReference[oaicite:3]{index=3}

---

## 🌍 NTU Interop WP1

Multimodal AI system for scalable real-world deployment.

---

## 👨‍💻 Author

Tajamul Ashraf — NTU

---

🚀 Happy building with Qwen3-VL!
EOF
