
# 🌟 NTU Interop WP1 — Qwen3-VL Multimodal Pipeline

> ⚡ High-performance multi-GPU vision-language pipeline for detection, reasoning, and OCR using Qwen3-VL

---

## 🧠 Overview

This project implements a **scalable multimodal inference pipeline** using **Qwen3-VL-8B-Instruct**.

It performs:

- 🔍 Object and semantic detection
- 🧍 Person localization
- 🔤 OCR (text extraction)
- ⚡ Multi-GPU parallel inference

---

## ⚙️ Installation

```bash
conda create -n qwen3vl_py311 python=3.11 -y
conda activate qwen3vl_py311

pip install torch==2.6.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install "transformers>=4.51.0" accelerate sentencepiece pillow tqdm
pip install flash-attn --no-build-isolation
pip install qwen-vl-utils==0.0.14
huggingface-cli login
```
For more details, refer to the official Qwen3-VL repository:  
https://github.com/QwenLM/Qwen3-VL  

This implementation follows the same setup and guidelines provided there.
---

## 🚀 Running the Pipeline

### 🖥️ Single GPU

```bash
python run.py
```

### ⚡ Multi-GPU Parallel

```bash
python run_parallel.py \
  --gpu-ids 0,4,6,7 \
  --batch-size 4 \
  --allow-tf32
```

---

## 📂 Project Structure

```text
Qwen3VL/
├── images/
├── qwen3vl_outputs/
├── qwen3vl_parallel_outputs_v2/
├── run.py
├── run_parallel.py
└── README.md
```

---

## 🔍 Pipeline

```text
Image → Resize → 3× Model Inference → JSON → Annotated Image
```

---

## 📦 Outputs

```text
image.json
image_annotated.png
```

```json
{
  "image_path": "images/example.jpg",
  "general": [...],
  "person": [...],
  "ocr": [...],
  "inference_time_sec": 9.54
}
```

---

## ⚠️ Batch Note

```python
for img in batch:
    process_image(img)
```

---

## 👨‍💻 Author

Tajamul Ashraf

---
