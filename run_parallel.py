import argparse
import json
import os
import subprocess
import sys
import time
from queue import Queue
import threading
from typing import List, Optional, Any

import torch
from PIL import Image, ImageDraw
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


# =========================================================
# CONFIG
# =========================================================
MODEL_NAME = "Qwen/Qwen3-VL-8B-Instruct"
INPUT_FOLDER = "/home2/tajamul/Qwen3VL/images"
OUTPUT_FOLDER = "./qwen3vl_parallel_outputs_v2"
DEFAULT_GPU_IDS = "0,4,6,7"
DEFAULT_BATCH_SIZE = 4
DEFAULT_ATTN = "auto"  # auto | flash_attention_2 | sdpa | none

TARGET_CATEGORIES = [
    "person", "face", "tattoo", "blood", "palm", "text", "weapon", "keyboard", "hotel room",
    "nude body parts like exposed breasts, genitals, or buttocks, anus, armpits, belly, feet",
    "school uniform", "school logos/badges",
]

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

model = None
processor = None


# =========================================================
# ARGUMENTS
# =========================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-folder", type=str, default=INPUT_FOLDER)
    parser.add_argument("--output-folder", type=str, default=OUTPUT_FOLDER)
    parser.add_argument("--gpu-ids", type=str, default=DEFAULT_GPU_IDS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--attn", type=str, default=DEFAULT_ATTN)
    parser.add_argument("--allow-tf32", action="store_true")
    parser.add_argument("--resize-max", type=int, default=1024)

    # internal worker args
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--worker-gpu", type=int, default=None)
    parser.add_argument("--worker-shard-file", type=str, default=None)
    return parser.parse_args()


# =========================================================
# HELPERS
# =========================================================
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def parse_gpu_ids(gpu_ids_text: str) -> List[int]:
    ids: List[int] = []
    for x in gpu_ids_text.split(","):
        x = x.strip()
        if x:
            ids.append(int(x))
    if not ids:
        raise ValueError("No valid GPU ids found.")
    return ids


def is_real_input_image(filename: str) -> bool:
    base = os.path.basename(filename)
    ext = os.path.splitext(base.lower())[1]
    if ext not in IMAGE_EXTS:
        return False
    if base.endswith("_annotated.png"):
        return False
    if "_resized" in base:
        return False
    return True


def resolve_attn(attn: str) -> Optional[str]:
    attn = (attn or "auto").strip().lower()
    if attn in {"none", "off", "null"}:
        return None
    if attn == "sdpa":
        return "sdpa"
    if attn == "flash_attention_2":
        return "flash_attention_2"
    if attn != "auto":
        raise ValueError(f"Unsupported --attn value: {attn}")

    try:
        import flash_attn  # noqa: F401
        return "flash_attention_2"
    except Exception:
        return "sdpa"


def load_image_rgb(image_path: str) -> Image.Image:
    with Image.open(image_path) as img:
        return img.convert("RGB")


def resize_image_in_memory(image_path: str, max_size: int = 1024) -> Image.Image:
    img = load_image_rgb(image_path)
    w, h = img.size

    if max(w, h) <= max_size:
        return img

    scale = max_size / max(w, h)
    new_w, new_h = int(w * scale), int(h * scale)
    return img.resize((new_w, new_h))


def build_messages(image_input: Any, prompt: str):
    return [{
        "role": "user",
        "content": [
            {"type": "image", "image": image_input},
            {"type": "text", "text": prompt},
        ],
    }]


# =========================================================
# MODEL
# =========================================================
def load_model(attn: str = "auto", allow_tf32: bool = False) -> None:
    global model, processor

    if model is not None and processor is not None:
        return

    print(f"[PID {os.getpid()}] Loading model...", flush=True)

    if torch.cuda.is_available():
        torch.cuda.set_device(0)

    if allow_tf32:
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    attn_impl = resolve_attn(attn)

    model_kwargs = {
        "torch_dtype": "auto",
        "device_map": {"": 0},
        "low_cpu_mem_usage": True,
    }
    if attn_impl is not None:
        model_kwargs["attn_implementation"] = attn_impl

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        **model_kwargs,
    )
    processor = AutoProcessor.from_pretrained(MODEL_NAME)

    print(
        f"[PID {os.getpid()}] Model loaded. CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '')}",
        flush=True,
    )


# =========================================================
# INFERENCE
# =========================================================
@torch.inference_mode()
def run_model(image_input: Any, prompt: str, max_new_tokens: int = 1024) -> str:
    messages = build_messages(image_input, prompt)

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )

    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)

    input_len = inputs["input_ids"].shape[1]
    outputs = outputs[:, input_len:]

    return processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()


# =========================================================
# PROMPTS
# =========================================================
def general_prompt() -> str:
    cat = ", ".join(TARGET_CATEGORIES)
    return f"""
Locate every visible instance in the image that belongs to these categories:
"{cat}"

Return JSON only with bbox_2d.
"""


def person_prompt() -> str:
    return """
Detect persons and return JSON.
"""


def ocr_prompt() -> str:
    return """
Extract all text and return JSON.
"""


# =========================================================
# JSON PARSE
# =========================================================
def extract_json(text: str):
    try:
        return json.loads(text)
    except Exception:
        try:
            start = text.find("[")
            end = text.rfind("]")
            if start != -1 and end != -1 and end > start:
                return json.loads(text[start:end + 1])
        except Exception:
            pass
        try:
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1 and end > start:
                return json.loads(text[start:end + 1])
        except Exception:
            pass
        return []

def merged_detections_for_drawing(gen_json, per_json, ocr_json):
    merged = []

    if isinstance(gen_json, list):
        for obj in gen_json:
            if not isinstance(obj, dict):
                continue
            if obj.get("label") == "person":
                continue
            merged.append(obj)

    if isinstance(per_json, list):
        merged.extend([obj for obj in per_json if isinstance(obj, dict)])

    if isinstance(ocr_json, list):
        merged.extend([obj for obj in ocr_json if isinstance(obj, dict)])

    return merged
# =========================================================
# DRAW
# =========================================================
def scale_bbox(bbox, w: int, h: int):
    x1, y1, x2, y2 = bbox
    return [
        int(x1 / 1000 * w),
        int(y1 / 1000 * h),
        int(x2 / 1000 * w),
        int(y2 / 1000 * h),
    ]


def draw_boxes(image_path: str, detections, output_path: str) -> None:
    with Image.open(image_path).convert("RGB") as img:
        draw = ImageDraw.Draw(img)
        w, h = img.size

        for obj in detections:
            if not isinstance(obj, dict) or "bbox_2d" not in obj:
                continue

            try:
                x1, y1, x2, y2 = scale_bbox(obj["bbox_2d"], w, h)
            except Exception:
                continue

            label = obj.get("label", "")
            draw.rectangle([x1, y1, x2, y2], outline="blue", width=3)
            draw.text((x1, max(0, y1 - 15)), str(label), fill="blue")

        img.save(output_path)


# =========================================================
# PROCESS ONE IMAGE
# =========================================================
def process_image(img_path: str, out_dir: str, resize_max: int) -> float:
    name = os.path.splitext(os.path.basename(img_path))[0]

    # resize only in memory, not saved to disk
    resized_img = resize_image_in_memory(img_path, max_size=resize_max)

    t0 = time.time()

    gen_raw = run_model(resized_img, general_prompt())
    per_raw = run_model(resized_img, person_prompt())
    ocr_raw = run_model(resized_img, ocr_prompt())

    inference_time = time.time() - t0

    gen_json = extract_json(gen_raw)
    per_json = extract_json(per_raw)
    ocr_json = extract_json(ocr_raw)

    result = {
        "image_path": img_path,
        "general": gen_json,
        "person": per_json,
        "ocr": ocr_json,
        "raw_outputs": {
            "general": gen_raw,
            "person": per_raw,
            "ocr": ocr_raw,
        },
        "inference_time_sec": round(inference_time, 3),
    }

    json_path = os.path.join(out_dir, f"{name}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    draw_boxes(
    img_path,
    merged_detections_for_drawing(gen_json, per_json, ocr_json),
    os.path.join(out_dir, f"{name}_annotated.png"),
    )

    resized_img.close()

    print(f"{os.path.basename(img_path)} | inference: {round(inference_time, 2)}s", flush=True)
    return inference_time


# =========================================================
# WORKER
# =========================================================
def worker(worker_gpu: int, shard_file: str, out_dir: str, batch_size: int, attn: str, allow_tf32: bool, resize_max: int) -> None:
    ensure_dir(out_dir)

    with open(shard_file, "r", encoding="utf-8") as f:
        images = json.load(f)

    load_model(attn=attn, allow_tf32=allow_tf32)

    q: Queue = Queue(maxsize=max(2, batch_size * 2))

    def producer():
        for img in images:
            q.put(img)
        q.put(None)

    threading.Thread(target=producer, daemon=True).start()

    total = 0.0
    processed = 0

    while True:
        batch: List[str] = []
        item = None

        while len(batch) < batch_size:
            item = q.get()
            if item is None:
                break
            batch.append(item)

        if not batch:
            break

        for img in batch:
            total += process_image(img, out_dir, resize_max)
            processed += 1

        if item is None:
            break

    summary = {
        "worker_gpu": worker_gpu,
        "num_images": processed,
        "total_inference_time_sec": round(total, 3),
        "average_inference_time_sec": round(total / processed, 3) if processed else 0.0,
    }

    summary_path = os.path.join(out_dir, f"_worker_gpu_{worker_gpu}_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[GPU {worker_gpu}] total inference: {round(total, 2)}s", flush=True)


# =========================================================
# MAIN
# =========================================================
def main() -> None:
    args = parse_args()
    ensure_dir(args.output_folder)

    if args.worker:
        if args.worker_gpu is None or args.worker_shard_file is None:
            raise ValueError("Worker mode requires --worker-gpu and --worker-shard-file")
        worker(
            worker_gpu=args.worker_gpu,
            shard_file=args.worker_shard_file,
            out_dir=args.output_folder,
            batch_size=max(1, args.batch_size),
            attn=args.attn,
            allow_tf32=args.allow_tf32,
            resize_max=args.resize_max,
        )
        return

    images = [
        os.path.join(args.input_folder, f)
        for f in sorted(os.listdir(args.input_folder))
        if is_real_input_image(f)
    ]

    gpu_ids = parse_gpu_ids(args.gpu_ids)

    shards = [[] for _ in gpu_ids]
    for i, img in enumerate(images):
        shards[i % len(gpu_ids)].append(img)

    shard_dir = os.path.join(args.output_folder, "_worker_shards")
    ensure_dir(shard_dir)

    total_start = time.time()
    procs = []

    for i, gpu in enumerate(gpu_ids):
        if not shards[i]:
            continue

        shard_file = os.path.join(shard_dir, f"gpu_{gpu}.json")
        with open(shard_file, "w", encoding="utf-8") as f:
            json.dump(shards[i], f)

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)

        p = subprocess.Popen(
            [
                sys.executable,
                __file__,
                "--worker",
                "--worker-gpu",
                str(gpu),
                "--worker-shard-file",
                shard_file,
                "--output-folder",
                args.output_folder,
                "--batch-size",
                str(max(1, args.batch_size)),
                "--attn",
                args.attn,
                "--resize-max",
                str(args.resize_max),
            ] + (["--allow-tf32"] if args.allow_tf32 else []),
            env=env,
        )
        procs.append(p)

    for p in procs:
        p.wait()

    total_time = time.time() - total_start

    run_summary = {
        "input_folder": args.input_folder,
        "output_folder": args.output_folder,
        "gpu_ids": gpu_ids,
        "num_images": len(images),
        "batch_size": max(1, args.batch_size),
        "attn": args.attn,
        "allow_tf32": bool(args.allow_tf32),
        "resize_max": args.resize_max,
        "total_wall_time_sec": round(total_time, 3),
    }

    with open(os.path.join(args.output_folder, "run_summary.json"), "w", encoding="utf-8") as f:
        json.dump(run_summary, f, indent=2)

    print("\n==============================")
    print(f"TOTAL TIME (ALL IMAGES): {round(total_time, 2)} sec")
    print("==============================")


if __name__ == "__main__":
    main()