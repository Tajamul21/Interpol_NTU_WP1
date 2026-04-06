# 🌟 NTU Interpol WP1 — Qwen3-VL Multimodal Pipeline

<p align="center">
  <a href="https://github.com/QwenLM/Qwen3-VL">
    <img src="https://img.shields.io/badge/Qwen3--VL-Official%20Repo-blue?style=for-the-badge" />
  </a>
  <a href="https://github.com/huggingface/transformers">
    <img src="https://img.shields.io/badge/HuggingFace-Transformers-yellow?style=for-the-badge" />
  </a>
  <img src="https://img.shields.io/badge/Python-3.11-green?style=for-the-badge" />
  <img src="https://img.shields.io/badge/PyTorch-2.6-red?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Multi--GPU-Supported-purple?style=for-the-badge" />
</p>

<p align="center">
  ⚡ High-performance multi-GPU vision-language pipeline for detection, reasoning, and OCR using Qwen3-VL
</p>

---

## 🧠 Overview

This project implements a **scalable multimodal inference pipeline** using **Qwen3-VL-8B-Instruct** for large-scale image understanding.

It supports:

- 🔍 **Object and semantic detection**
- 🧍 **Person localization**
- 🔤 **OCR / text extraction**
- ⚡ **Multi-GPU parallel inference**
- 📦 **Structured JSON outputs**
- 🖼️ **Annotated image generation**

Designed for **efficient large-scale processing**, this pipeline enables robust multimodal reasoning over visual data with clean and reproducible outputs.

---

## ✨ Features

- High-throughput **multi-GPU execution**
- Separate reasoning stages for **general detection**, **person detection**, and **OCR**
- Structured outputs in **JSON format**
- Easy-to-run **single GPU** and **parallel GPU** modes
- Compatible with **Qwen3-VL** ecosystem and setup
- Built for research and scalable deployment workflows

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

> [!NOTE]
> For more details, refer to the official Qwen3-VL repository.

<p>
  <a href="https://github.com/QwenLM/Qwen3-VL">
    <img src="https://img.shields.io/badge/🔗%20Qwen3--VL%20Official%20Repo-Visit-blue?style=for-the-badge" />
  </a>
</p>

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
├── images/                          # Input images
├── qwen3vl_outputs/                 # Single GPU outputs
├── qwen3vl_parallel_outputs_v2/     # Multi-GPU outputs
├── run.py                           # Single GPU pipeline
├── run_parallel.py                  # Parallel pipeline
└── README.md
```

---

## 🔍 What the Pipeline Does

Each image goes through three reasoning stages:

- 🧠 **General Detection** → objects, scene context, and semantic categories
- 🧍 **Person Detection** → human localization only
- 🔤 **OCR Extraction** → visible text in the image

Each stage returns structured results with bounding boxes and relevant metadata.

---

## 🧩 Pipeline Flow

### Single GPU

```text
Image → Resize → 3× Model Inference → JSON Output → Annotated Image
```

### Multi-GPU

```text
Images → Shard Across GPUs → Spawn Workers → Run Inference → Save Outputs
```

### Parallel Execution Details

- One worker is assigned per GPU
- Each worker loads its own model instance
- Images are distributed across devices
- Processing happens in parallel for improved throughput

---

## 📦 Outputs

For each image, the pipeline produces:

```text
image.json
image_annotated.png
```

### Example JSON Output

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

## ⚙️ Key Arguments

| Argument | Description |
|----------|-------------|
| `--gpu-ids` | GPUs to use for parallel execution |
| `--batch-size` | Queue grouping size, not true model batching |
| `--allow-tf32` | Enables faster computation on supported hardware |
| `--resize-max` | Maximum input image resolution |
| `--attn` | Attention implementation mode |

---

## ⚡ Performance Notes

- Each image requires **3 model calls**
- Each GPU loads a **separate model copy**
- Multi-GPU improves **throughput**, not single-image latency
- Best results come from balancing **GPU count**, **image size**, and **storage speed**

### Example Runtime

```text
[GPU 0] total inference: 47.33s
[GPU 4] total inference: 32.99s
[GPU 6] total inference: 60.30s
[GPU 7] total inference: 26.52s

TOTAL TIME: 142.91 sec
```

👉 Average per-image inference time, excluding model load: **~12.9 seconds**

---

## ⚠️ Important Note on `--batch-size`

`--batch-size` does **not** mean true model batching.

It only groups images at the queue level:

```python
for img in batch:
    process_image(img)
```

So the model still processes **one image at a time**.

---

## 🎯 Best Practices

- Use **multiple GPUs** for large datasets
- Enable **`--allow-tf32`** when supported
- Keep image size **≤ 1024** for better speed-memory balance
- Use **SSD storage** for faster read/write throughput
- Prefer **Flash Attention** on modern GPUs for better efficiency

---

## 🖥️ Hardware Requirements

- GPU recommended with **at least 16 GB VRAM**
- CUDA **11+**
- Ampere or newer GPUs recommended for best Flash Attention performance

---

## 🌍 Project Context

This work is part of:

### 🎓 NTU Interoperability Work Package 1 (WP1)

**Focus:** Scalable Multimodal AI Systems

---

## 🙏 Acknowledgements

This project builds upon and is inspired by the excellent open-source work of:

- **Qwen Team** for the **Qwen3-VL** model and ecosystem
- **Hugging Face** for the Transformers framework
- **PyTorch** for the deep learning infrastructure
- The broader multimodal research community advancing large vision-language models

Special thanks to the contributors and maintainers of the tools and libraries that made this pipeline possible.

---

## 👨‍💻 Author

**Tajamul Ashraf**  
Multimodal AI Research — NTU

---

## 🔗 Useful Links

<p>
  <a href="https://github.com/QwenLM/Qwen3-VL">
    <img src="https://img.shields.io/badge/Qwen3--VL-GitHub-blue?style=for-the-badge" />
  </a>
  <a href="https://github.com/huggingface/transformers">
    <img src="https://img.shields.io/badge/Transformers-HuggingFace-yellow?style=for-the-badge" />
  </a>
  <a href="https://pytorch.org/">
    <img src="https://img.shields.io/badge/PyTorch-Official-red?style=for-the-badge" />
  </a>
</p>

---

<p align="center">
  🚀 Happy building with Qwen3-VL!
</p>
