import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


# =========================================================
# INTERPOL WP1
# =========================================================
MODEL_NAME = "Qwen/Qwen3-VL-8B-Instruct"
TORCH_DTYPE = "auto"
MAX_NEW_TOKENS_DET = 1024
MAX_NEW_TOKENS_OCR = 2048

INPUT_FOLDER = "/home2/tajamul/Qwen3VL/images"
OUTPUT_FOLDER = "./qwen3vl_batch_outputs"
DEFAULT_GPU_IDS = "0,4,6,7"
DEFAULT_BATCH_SIZE = 2
DEFAULT_ATTN = "auto"  # auto | sdpa | flash_attention_2 | none

TARGET_CATEGORIES = [
    "person",
    "face",
    "tattoo",
    "blood",
    "palm",
    "text",
    "weapon",
    "keyboard",
    "hotel room",
]

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# =========================================================
# Process-local globals
# =========================================================
model = None
processor = None


# =========================================================
# Helpers
# =========================================================
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def write_json(path: str, data: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def is_image_file(filename: str) -> bool:
    return os.path.splitext(filename.lower())[1] in IMAGE_EXTS


def round_num(value: Optional[float], ndigits: int = 3) -> Optional[float]:
    if value is None:
        return None
    return round(float(value), ndigits)


def parse_gpu_ids(gpu_ids_text: str) -> List[int]:
    ids: List[int] = []
    for item in gpu_ids_text.split(","):
        item = item.strip()
        if not item:
            continue
        ids.append(int(item))
    if not ids:
        raise ValueError("No GPU ids were provided.")
    return ids


def load_pil_image(image_path: str) -> Image.Image:
    with Image.open(image_path) as im:
        return im.convert("RGB")


def get_image_area(image_path: str) -> int:
    try:
        with Image.open(image_path) as im:
            w, h = im.size
        return int(w) * int(h)
    except Exception:
        return 0


def list_image_records(input_folder: str) -> List[Dict[str, Any]]:
    if not os.path.isdir(input_folder):
        return []

    image_files = [
        os.path.join(input_folder, f)
        for f in sorted(os.listdir(input_folder))
        if is_image_file(f)
    ]

    records = []
    for image_path in image_files:
        records.append({
            "image_path": image_path,
            "area": get_image_area(image_path),
        })
    return records


def greedy_balance_by_area(image_records: List[Dict[str, Any]], num_shards: int) -> List[List[Dict[str, Any]]]:
    shards: List[List[Dict[str, Any]]] = [[] for _ in range(num_shards)]
    loads: List[int] = [0 for _ in range(num_shards)]

    items = sorted(image_records, key=lambda x: int(x.get("area", 0)), reverse=True)
    for record in items:
        shard_idx = min(range(num_shards), key=lambda i: loads[i])
        shards[shard_idx].append(record)
        loads[shard_idx] += max(1, int(record.get("area", 0)))

    return shards


def chunk_list(items: List[Any], chunk_size: int) -> List[List[Any]]:
    return [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]


def resolve_attn_implementation(requested: str) -> Optional[str]:
    requested = (requested or "auto").strip().lower()

    if requested in {"none", "null", "off"}:
        return None
    if requested == "sdpa":
        return "sdpa"
    if requested == "flash_attention_2":
        return "flash_attention_2"
    if requested != "auto":
        raise ValueError(f"Unsupported --attn value: {requested}")

    try:
        import flash_attn  # noqa: F401
        return "flash_attention_2"
    except Exception:
        return "sdpa"


# =========================================================
# Load model + processor
# =========================================================
def load_model(attn: str, allow_tf32: bool = False) -> None:
    global model, processor

    if model is not None and processor is not None:
        return

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    if torch.cuda.is_available():
        torch.cuda.set_device(0)

    if allow_tf32:
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    attn_impl = resolve_attn_implementation(attn)

    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "ALL")
    print(
        f"Loading model... CUDA_VISIBLE_DEVICES={visible} attn={attn_impl} allow_tf32={allow_tf32}",
        flush=True,
    )

    model_kwargs: Dict[str, Any] = {
        "torch_dtype": TORCH_DTYPE,
        "device_map": "auto",
        "low_cpu_mem_usage": True,
    }
    if attn_impl is not None:
        model_kwargs["attn_implementation"] = attn_impl

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        **model_kwargs,
    )
    model.eval()

    processor = AutoProcessor.from_pretrained(MODEL_NAME)

    print(
        f"Model loaded. model.device={getattr(model, 'device', 'unknown')}",
        flush=True,
    )


# =========================================================
# Messages + inference
# =========================================================
def build_messages(image_input: Any, prompt: str) -> List[Dict[str, Any]]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_input},
                {"type": "text", "text": prompt},
            ],
        }
    ]


@torch.inference_mode()
def run_qwen3_vl_batch(
    image_inputs: List[Any],
    prompt: str,
    max_new_tokens: int,
) -> Tuple[List[str], Dict[str, Optional[float]]]:
    if model is None or processor is None:
        raise RuntimeError("Model and processor are not loaded.")

    batch_size = len(image_inputs)
    total_wall_start = time.perf_counter()

    batch_messages = [build_messages(img, prompt) for img in image_inputs]

    preprocess_start = time.perf_counter()
    inputs = processor.apply_chat_template(
        batch_messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
        padding=True,
    )
    inputs.pop("token_type_ids", None)
    inputs = inputs.to(model.device)
    preprocess_wall_ms = (time.perf_counter() - preprocess_start) * 1000.0

    generate_gpu_ms: Optional[float] = None
    generate_wall_start = time.perf_counter()

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
        )
        end_event.record()
        torch.cuda.synchronize()
        generate_gpu_ms = float(start_event.elapsed_time(end_event))
    else:
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
        )

    generate_wall_ms = (time.perf_counter() - generate_wall_start) * 1000.0

    decode_start = time.perf_counter()
    input_ids = inputs["input_ids"]
    generated_ids_trimmed = [
        out_ids[len(in_ids):].cpu()
        for in_ids, out_ids in zip(input_ids, generated_ids)
    ]
    output_texts = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    decode_wall_ms = (time.perf_counter() - decode_start) * 1000.0

    total_wall_ms = (time.perf_counter() - total_wall_start) * 1000.0

    timing = {
        "batch_size": batch_size,
        "preprocess_wall_ms": round_num(preprocess_wall_ms),
        "generate_gpu_ms": round_num(generate_gpu_ms),
        "generate_wall_ms": round_num(generate_wall_ms),
        "decode_wall_ms": round_num(decode_wall_ms),
        "total_wall_ms": round_num(total_wall_ms),
        "amortized_total_wall_ms": round_num(total_wall_ms / max(1, batch_size)),
        "amortized_generate_gpu_ms": round_num(
            (generate_gpu_ms / max(1, batch_size)) if generate_gpu_ms is not None else None
        ),
    }

    return [text.strip() for text in output_texts], timing


# =========================================================
# JSON extraction
# =========================================================
def strip_markdown_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```json"):
        text = text[len("```json"):].strip()
    elif text.startswith("```"):
        text = text[len("```"):].strip()

    if text.endswith("```"):
        text = text[:-3].strip()

    return text


def extract_json_from_text(text: str) -> Union[List[Any], Dict[str, Any], None]:
    text = strip_markdown_fence(text)

    try:
        return json.loads(text)
    except Exception:
        pass

    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1 and end > start:
        chunk = text[start:end + 1]
        try:
            return json.loads(chunk)
        except Exception:
            pass

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        chunk = text[start:end + 1]
        try:
            return json.loads(chunk)
        except Exception:
            pass

    return None


# =========================================================
# Geometry + drawing
# =========================================================
def scale_bbox_1000_to_pixels(bbox: List[float], width: int, height: int) -> List[int]:
    if len(bbox) != 4:
        raise ValueError(f"Invalid bbox length: {bbox}")

    x1, y1, x2, y2 = bbox

    x1 = int(round((x1 / 1000.0) * width))
    y1 = int(round((y1 / 1000.0) * height))
    x2 = int(round((x2 / 1000.0) * width))
    y2 = int(round((y2 / 1000.0) * height))

    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(0, min(width - 1, x2))
    y2 = max(0, min(height - 1, y2))

    if x1 > x2:
        x1, x2 = x2, x1
    if y1 > y2:
        y1, y2 = y2, y1

    return [x1, y1, x2, y2]


def get_font(size: int = 16) -> Optional[ImageFont.ImageFont]:
    for font_name in ["DejaVuSans.ttf", "Arial.ttf"]:
        try:
            return ImageFont.truetype(font_name, size=size)
        except Exception:
            pass
    try:
        return ImageFont.load_default()
    except Exception:
        return None


# =========================================================
# Normalization
# =========================================================
def normalize_general_detection_output(
    parsed: Union[List[Any], Dict[str, Any], None]
) -> List[Dict[str, Any]]:
    if parsed is None:
        return []

    if isinstance(parsed, dict):
        if isinstance(parsed.get("objects"), list):
            items = parsed["objects"]
        else:
            items = [parsed]
    elif isinstance(parsed, list):
        items = parsed
    else:
        return []

    normalized = []
    for item in items:
        if not isinstance(item, dict):
            continue

        bbox = item.get("bbox_2d") or item.get("bbox") or item.get("box")
        if not isinstance(bbox, list) or len(bbox) != 4:
            continue

        label = item.get("label") or item.get("category") or item.get("name") or ""
        text_content = item.get("text_content", "")

        normalized.append({
            "label": str(label).strip(),
            "bbox_2d": bbox,
            "text_content": str(text_content).strip() if text_content is not None else "",
        })

    return normalized


def normalize_person_output(
    parsed: Union[List[Any], Dict[str, Any], None]
) -> List[Dict[str, Any]]:
    if parsed is None:
        return []

    if isinstance(parsed, dict):
        if isinstance(parsed.get("persons"), list):
            items = parsed["persons"]
        else:
            items = [parsed]
    elif isinstance(parsed, list):
        items = parsed
    else:
        return []

    normalized = []
    for item in items:
        if not isinstance(item, dict):
            continue

        person_bbox = item.get("bbox_2d") or item.get("person_bbox_2d") or item.get("bbox")
        if not isinstance(person_bbox, list) or len(person_bbox) != 4:
            continue

        face_bbox = item.get("face_bbox_2d")
        if not (isinstance(face_bbox, list) and len(face_bbox) == 4):
            face_bbox = None

        action = item.get("action", "")
        clothing = item.get("visible_clothing", [])
        accessories = item.get("visible_accessories", [])
        tattoo_detected = item.get("tattoo_detected", False)
        blood_visible = item.get("blood_visible", False)
        palm_visible = item.get("palm_visible", False)
        weapon_near_person = item.get("weapon_near_person", False)

        if not isinstance(clothing, list):
            clothing = [str(clothing)]
        if not isinstance(accessories, list):
            accessories = [str(accessories)]

        normalized.append({
            "label": "person",
            "bbox_2d": person_bbox,
            "face_bbox_2d": face_bbox,
            "action": str(action).strip(),
            "gender": "",
            "ethnicity": "",
            "age": "",
            "visible_clothing": [str(x).strip() for x in clothing if str(x).strip()],
            "visible_accessories": [str(x).strip() for x in accessories if str(x).strip()],
            "tattoo_detected": bool(tattoo_detected),
            "blood_visible": bool(blood_visible),
            "palm_visible": bool(palm_visible),
            "weapon_near_person": bool(weapon_near_person),
        })

    return normalized


def normalize_ocr_output(
    parsed: Union[List[Any], Dict[str, Any], None]
) -> List[Dict[str, Any]]:
    if parsed is None:
        return []

    if isinstance(parsed, dict):
        if isinstance(parsed.get("texts"), list):
            items = parsed["texts"]
        elif isinstance(parsed.get("objects"), list):
            items = parsed["objects"]
        else:
            items = [parsed]
    elif isinstance(parsed, list):
        items = parsed
    else:
        return []

    normalized = []
    for item in items:
        if not isinstance(item, dict):
            continue

        bbox = item.get("bbox_2d") or item.get("bbox") or item.get("box")
        if not isinstance(bbox, list) or len(bbox) != 4:
            continue

        text_content = item.get("text_content") or item.get("text") or item.get("label") or ""

        normalized.append({
            "label": "text",
            "bbox_2d": bbox,
            "text_content": str(text_content).strip(),
        })

    return normalized


# =========================================================
# Prompts
# =========================================================
def build_general_detection_prompt(categories: List[str]) -> str:
    cat_str = ", ".join(categories)
    return f'''
Locate every visible instance in the image that belongs to these categories:
"{cat_str}"

Rules:
- Return JSON only.
- Use relative coordinates scaled from 0 to 1000.
- If an item is not present, do not include it.
- Allowed labels are only: person, face, tattoo, blood, palm, text, weapon, keyboard, hotel room.
- For text regions, use label "text".
- For weapon-like objects, use label "weapon".
- For visible hotel room scenes or obvious hotel-room context, use label "hotel room".

Output format:
[
  {{
    "label": "category_name",
    "bbox_2d": [x1, y1, x2, y2]
  }}
]
'''.strip()


def build_person_prompt() -> str:
    return '''
Detect all visible persons in the image.

For each detected person, return JSON only in this format:
[
  {
    "label": "person",
    "bbox_2d": [x1, y1, x2, y2],
    "face_bbox_2d": [x1, y1, x2, y2],
    "action": "short visible action if clear, else empty string",
    "visible_clothing": ["item1", "item2"],
    "visible_accessories": ["item1", "item2"],
    "tattoo_detected": true,
    "blood_visible": false,
    "palm_visible": true,
    "weapon_near_person": false
  }
]

Rules:
- Use relative coordinates scaled from 0 to 1000.
- Include face_bbox_2d only if a face is visible.
- visible_clothing and visible_accessories must be arrays.
- tattoo_detected on body, blood_visible, palm_visible, weapon_near_person must be true or false.
- Return JSON only, no markdown.
'''.strip()


def build_ocr_prompt() -> str:
    return '''
Spot all readable text in the image and return JSON only.

Rules:
- Use relative coordinates scaled from 0 to 1000.
- Return line-level text regions.
- If no text is visible, return [].
- Do not return markdown.

Output format:
[
  {
    "label": "text",
    "bbox_2d": [x1, y1, x2, y2],
    "text_content": "recognized text"
  }
]
'''.strip()


# =========================================================
# Drawing
# =========================================================
def flatten_for_drawing(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    draw_items: List[Dict[str, Any]] = []

    for item in result.get("general_detections", []):
        label = item.get("label", "")
        if label in {"face", "person"}:
            continue
        draw_items.append({
            "label": label,
            "bbox_2d": item.get("bbox_2d"),
            "text_content": item.get("text_content", ""),
        })

    for person_item in result.get("persons", []):
        draw_items.append({
            "label": "person",
            "bbox_2d": person_item.get("bbox_2d"),
            "text_content": person_item.get("action", ""),
        })

        face_bbox = person_item.get("face_bbox_2d")
        if face_bbox:
            draw_items.append({
                "label": "face",
                "bbox_2d": face_bbox,
                "text_content": "",
            })

    for item in result.get("ocr_texts", []):
        draw_items.append({
            "label": "text",
            "bbox_2d": item.get("bbox_2d"),
            "text_content": item.get("text_content", ""),
        })

    return draw_items


def draw_bboxes_from_pil(
    source_image: Image.Image,
    result: Dict[str, Any],
    output_path: str,
) -> str:
    image = source_image.copy()
    draw = ImageDraw.Draw(image)
    width, height = image.size
    font = get_font(16)

    draw_items = flatten_for_drawing(result)

    for item in draw_items:
        bbox = item.get("bbox_2d")
        if not bbox:
            continue

        try:
            x1, y1, x2, y2 = scale_bbox_1000_to_pixels(bbox, width, height)
        except Exception:
            continue

        label = item.get("label", "").strip()
        text_content = item.get("text_content", "").strip()

        draw.rectangle([x1, y1, x2, y2], outline="blue", width=3)

        caption = label
        if text_content:
            caption = f"{label}: {text_content}" if label else text_content

        if caption:
            text_y = max(0, y1 - 18)
            draw.text((x1, text_y), caption, fill="blue", font=font)

    image.save(output_path)
    return output_path


# =========================================================
# Batched per-image pipeline
# =========================================================
def process_one_batch(
    batch_records: List[Dict[str, Any]],
    output_folder: str,
    categories: List[str],
    worker_tag: str = "",
) -> List[Dict[str, Any]]:
    batch_start = time.perf_counter()

    batch_records = sorted(batch_records, key=lambda x: int(x.get("area", 0)), reverse=True)
    image_paths = [item["image_path"] for item in batch_records]
    pil_images = [load_pil_image(path) for path in image_paths]

    print(
        f"{worker_tag} Batch start | batch_size={len(batch_records)} | "
        f"images={[os.path.basename(p) for p in image_paths]}",
        flush=True,
    )

    general_prompt = build_general_detection_prompt(categories)
    person_prompt = build_person_prompt()
    ocr_prompt = build_ocr_prompt()

    general_raws, general_timing = run_qwen3_vl_batch(
        image_inputs=pil_images,
        prompt=general_prompt,
        max_new_tokens=MAX_NEW_TOKENS_DET,
    )
    general_items_list = [
        normalize_general_detection_output(extract_json_from_text(text))
        for text in general_raws
    ]

    person_raws, person_timing = run_qwen3_vl_batch(
        image_inputs=pil_images,
        prompt=person_prompt,
        max_new_tokens=MAX_NEW_TOKENS_DET,
    )
    persons_list = [
        normalize_person_output(extract_json_from_text(text))
        for text in person_raws
    ]

    ocr_raws, ocr_timing = run_qwen3_vl_batch(
        image_inputs=pil_images,
        prompt=ocr_prompt,
        max_new_tokens=MAX_NEW_TOKENS_OCR,
    )
    ocr_items_list = [
        normalize_ocr_output(extract_json_from_text(text))
        for text in ocr_raws
    ]

    shared_inference_latency_ms = (
        (general_timing.get("total_wall_ms") or 0.0) +
        (person_timing.get("total_wall_ms") or 0.0) +
        (ocr_timing.get("total_wall_ms") or 0.0)
    )
    amortized_inference_ms = (
        (general_timing.get("amortized_total_wall_ms") or 0.0) +
        (person_timing.get("amortized_total_wall_ms") or 0.0) +
        (ocr_timing.get("amortized_total_wall_ms") or 0.0)
    )
    shared_generate_gpu_ms = (
        (general_timing.get("generate_gpu_ms") or 0.0) +
        (person_timing.get("generate_gpu_ms") or 0.0) +
        (ocr_timing.get("generate_gpu_ms") or 0.0)
    )
    amortized_generate_gpu_ms = (
        (general_timing.get("amortized_generate_gpu_ms") or 0.0) +
        (person_timing.get("amortized_generate_gpu_ms") or 0.0) +
        (ocr_timing.get("amortized_generate_gpu_ms") or 0.0)
    )

    results: List[Dict[str, Any]] = []

    for idx, image_path in enumerate(image_paths):
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        annotated_path = os.path.join(output_folder, f"{base_name}_annotated.png")
        json_path = os.path.join(output_folder, f"{base_name}.json")

        result = {
            "status": "ok",
            "image_path": image_path,
            "categories_requested": categories,
            "batch_info": {
                "batch_size": len(batch_records),
                "area": batch_records[idx].get("area", 0),
                "images_in_batch": image_paths,
            },
            "general_detections": general_items_list[idx],
            "persons": persons_list[idx],
            "ocr_texts": ocr_items_list[idx],
            "timings_ms": {
                "general_detection": general_timing,
                "person_detection": person_timing,
                "ocr": ocr_timing,
                "inference_latency_ms": round_num(shared_inference_latency_ms),
                "amortized_inference_ms": round_num(amortized_inference_ms),
                "generate_gpu_latency_ms": round_num(shared_generate_gpu_ms),
                "amortized_generate_gpu_ms": round_num(amortized_generate_gpu_ms),
                "image_total_wall_ms": None,
            },
            "raw_responses": {
                "general_detection": general_raws[idx],
                "person_detection": person_raws[idx],
                "ocr": ocr_raws[idx],
            },
        }

        draw_bboxes_from_pil(
            source_image=pil_images[idx],
            result=result,
            output_path=annotated_path,
        )

        result["timings_ms"]["image_total_wall_ms"] = round_num(
            (time.perf_counter() - batch_start) * 1000.0
        )

        write_json(json_path, result)
        results.append(result)

    batch_total_wall_ms = (time.perf_counter() - batch_start) * 1000.0
    print(
        f"{worker_tag} Batch done | batch_size={len(batch_records)} | "
        f"batch_total={batch_total_wall_ms / 1000.0:.3f}s | "
        f"inference_latency={shared_inference_latency_ms / 1000.0:.3f}s | "
        f"amortized_per_image={amortized_inference_ms / 1000.0:.3f}s",
        flush=True,
    )

    return results


def process_batch_recursive(
    batch_records: List[Dict[str, Any]],
    output_folder: str,
    categories: List[str],
    worker_tag: str = "",
) -> List[Dict[str, Any]]:
    try:
        return process_one_batch(
            batch_records=batch_records,
            output_folder=output_folder,
            categories=categories,
            worker_tag=worker_tag,
        )
    except RuntimeError as e:
        message = str(e).lower()
        if "out of memory" in message and len(batch_records) > 1:
            print(
                f"{worker_tag} CUDA OOM on batch_size={len(batch_records)}. Splitting batch.",
                flush=True,
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            mid = len(batch_records) // 2
            left = process_batch_recursive(
                batch_records=batch_records[:mid],
                output_folder=output_folder,
                categories=categories,
                worker_tag=worker_tag,
            )
            right = process_batch_recursive(
                batch_records=batch_records[mid:],
                output_folder=output_folder,
                categories=categories,
                worker_tag=worker_tag,
            )
            return left + right
        raise


def process_record_list(
    image_records: List[Dict[str, Any]],
    output_folder: str,
    categories: List[str],
    batch_size: int,
    worker_tag: str = "",
) -> List[Dict[str, Any]]:
    ensure_dir(output_folder)

    if not image_records:
        print(f"{worker_tag} No images assigned to this worker.", flush=True)
        return []

    ordered = sorted(image_records, key=lambda x: int(x.get("area", 0)), reverse=True)
    record_batches = chunk_list(ordered, max(1, batch_size))

    all_results: List[Dict[str, Any]] = []
    for batch_records in record_batches:
        try:
            results = process_batch_recursive(
                batch_records=batch_records,
                output_folder=output_folder,
                categories=categories,
                worker_tag=worker_tag,
            )
            all_results.extend(results)
        except Exception as e:
            for record in batch_records:
                image_path = record["image_path"]
                base_name = os.path.splitext(os.path.basename(image_path))[0]
                json_path = os.path.join(output_folder, f"{base_name}.json")
                error_result = {
                    "status": "failed",
                    "image_path": image_path,
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                    "timings_ms": {
                        "general_detection": {},
                        "person_detection": {},
                        "ocr": {},
                        "inference_latency_ms": None,
                        "amortized_inference_ms": None,
                        "generate_gpu_latency_ms": None,
                        "amortized_generate_gpu_ms": None,
                        "image_total_wall_ms": None,
                    },
                }
                write_json(json_path, error_result)
                all_results.append(error_result)
                print(f"{worker_tag} Failed on {image_path}: {e}", flush=True)

    return all_results


# =========================================================
# Worker summary + parent aggregation
# =========================================================
def build_worker_summary(
    worker_index: int,
    physical_gpu_id: int,
    image_records: List[Dict[str, Any]],
    results: List[Dict[str, Any]],
    model_load_wall_s: float,
    worker_total_wall_s: float,
) -> Dict[str, Any]:
    success_results = [r for r in results if r.get("status") == "ok"]
    failed_results = [r for r in results if r.get("status") != "ok"]

    worker_inference_wall_s_estimate = sum(
        (r.get("timings_ms", {}).get("amortized_inference_ms") or 0.0)
        for r in success_results
    ) / 1000.0

    return {
        "worker_index": worker_index,
        "physical_gpu_id": physical_gpu_id,
        "visible_cuda_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "images_assigned": len(image_records),
        "images_succeeded": len(success_results),
        "images_failed": len(failed_results),
        "model_load_wall_s": round_num(model_load_wall_s),
        "worker_total_wall_s": round_num(worker_total_wall_s),
        "worker_inference_wall_s_estimate": round_num(worker_inference_wall_s_estimate),
        "throughput_images_per_s": round_num(
            len(success_results) / worker_total_wall_s if worker_total_wall_s > 0 else 0.0
        ),
        "per_image": [
            {
                "image_path": r.get("image_path"),
                "status": r.get("status"),
                "inference_latency_ms": r.get("timings_ms", {}).get("inference_latency_ms"),
                "amortized_inference_ms": r.get("timings_ms", {}).get("amortized_inference_ms"),
                "image_total_wall_ms": r.get("timings_ms", {}).get("image_total_wall_ms"),
            }
            for r in results
        ],
    }


def run_worker_mode(
    worker_index: int,
    worker_gpu: int,
    input_list_json: str,
    output_folder: str,
    batch_size: int,
    attn: str,
    allow_tf32: bool,
) -> None:
    ensure_dir(output_folder)

    image_records = read_json(input_list_json)
    if not isinstance(image_records, list):
        raise ValueError(f"Invalid worker input list in {input_list_json}")

    worker_tag = f"[worker={worker_index} gpu={worker_gpu}]"
    print(
        f"{worker_tag} Starting worker. assigned_images={len(image_records)} "
        f"batch_size={batch_size} visible_cuda_devices={os.environ.get('CUDA_VISIBLE_DEVICES', 'ALL')}",
        flush=True,
    )

    worker_start = time.perf_counter()

    load_start = time.perf_counter()
    load_model(attn=attn, allow_tf32=allow_tf32)
    model_load_wall_s = time.perf_counter() - load_start

    results = process_record_list(
        image_records=image_records,
        output_folder=output_folder,
        categories=TARGET_CATEGORIES,
        batch_size=batch_size,
        worker_tag=worker_tag,
    )

    worker_total_wall_s = time.perf_counter() - worker_start
    worker_summary = build_worker_summary(
        worker_index=worker_index,
        physical_gpu_id=worker_gpu,
        image_records=image_records,
        results=results,
        model_load_wall_s=model_load_wall_s,
        worker_total_wall_s=worker_total_wall_s,
    )

    worker_summary_path = os.path.join(output_folder, f"_worker_{worker_index}_summary.json")
    write_json(worker_summary_path, worker_summary)

    print(
        f"{worker_tag} Done. model_load={model_load_wall_s:.3f}s | "
        f"worker_total={worker_total_wall_s:.3f}s | summary={worker_summary_path}",
        flush=True,
    )


def launch_worker_subprocess(
    script_path: str,
    worker_index: int,
    physical_gpu_id: int,
    image_records: List[Dict[str, Any]],
    output_folder: str,
    batch_size: int,
    attn: str,
    allow_tf32: bool,
) -> subprocess.Popen:
    input_list_json = os.path.join(output_folder, f"_worker_{worker_index}_inputs.json")
    write_json(input_list_json, image_records)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(physical_gpu_id)

    cmd = [
        sys.executable,
        script_path,
        "--worker-gpu", str(physical_gpu_id),
        "--worker-index", str(worker_index),
        "--input-list-json", input_list_json,
        "--output-folder", output_folder,
        "--batch-size", str(batch_size),
        "--attn", attn,
    ]
    if allow_tf32:
        cmd.append("--allow-tf32")

    print(
        f"[parent] Launching worker={worker_index} on physical_gpu={physical_gpu_id} "
        f"with {len(image_records)} images | batch_size={batch_size}",
        flush=True,
    )

    return subprocess.Popen(cmd, env=env)


def aggregate_final_outputs(
    image_records: List[Dict[str, Any]],
    output_folder: str,
    gpu_ids_used: List[int],
    total_wall_s: float,
) -> Dict[str, Any]:
    all_results: List[Dict[str, Any]] = []
    per_image: List[Dict[str, Any]] = []
    failed_images: List[str] = []

    for record in image_records:
        image_path = record["image_path"]
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        json_path = os.path.join(output_folder, f"{base_name}.json")

        if not os.path.exists(json_path):
            failed_images.append(image_path)
            continue

        item = read_json(json_path)
        all_results.append(item)

        timings = item.get("timings_ms", {})
        per_image.append({
            "image_path": item.get("image_path", image_path),
            "status": item.get("status", "unknown"),
            "inference_latency_ms": timings.get("inference_latency_ms"),
            "amortized_inference_ms": timings.get("amortized_inference_ms"),
            "image_total_wall_ms": timings.get("image_total_wall_ms"),
        })

        if item.get("status") != "ok":
            failed_images.append(image_path)

    all_results = sorted(all_results, key=lambda x: x.get("image_path", ""))
    per_image = sorted(per_image, key=lambda x: x.get("image_path", ""))

    summary_path = os.path.join(output_folder, "summary.json")
    write_json(summary_path, all_results)

    worker_summaries: List[Dict[str, Any]] = []
    worker_index = 0
    while True:
        worker_summary_path = os.path.join(output_folder, f"_worker_{worker_index}_summary.json")
        if not os.path.exists(worker_summary_path):
            break
        worker_summaries.append(read_json(worker_summary_path))
        worker_index += 1

    success_images = [x for x in per_image if x.get("status") == "ok"]
    total_amortized_inference_s = sum(
        (x.get("amortized_inference_ms") or 0.0) for x in success_images
    ) / 1000.0

    run_summary = {
        "input_folder": INPUT_FOLDER,
        "output_folder": output_folder,
        "num_images_total": len(image_records),
        "num_images_succeeded": len(success_images),
        "num_images_failed": len(per_image) - len(success_images),
        "gpu_ids_used": gpu_ids_used,
        "total_wall_s": round_num(total_wall_s),
        "total_amortized_inference_s": round_num(total_amortized_inference_s),
        "throughput_images_per_s": round_num(
            len(success_images) / total_wall_s if total_wall_s > 0 else 0.0
        ),
        "per_image": per_image,
        "worker_summaries": worker_summaries,
        "failed_images": failed_images,
    }

    run_summary_path = os.path.join(output_folder, "run_summary.json")
    write_json(run_summary_path, run_summary)

    print(f"[parent] Saved folder summary: {summary_path}", flush=True)
    print(f"[parent] Saved run summary: {run_summary_path}", flush=True)

    return run_summary


def run_parent_mode(
    input_folder: str,
    output_folder: str,
    gpu_ids: List[int],
    batch_size: int,
    attn: str,
    allow_tf32: bool,
) -> None:
    ensure_dir(output_folder)

    image_records = list_image_records(input_folder)
    if not image_records:
        print(f"No images found in: {input_folder}", flush=True)
        return

    shards = greedy_balance_by_area(image_records, len(gpu_ids))
    active = [
        (idx, gpu_ids[idx], shard)
        for idx, shard in enumerate(shards)
        if shard
    ]

    print(
        f"[parent] Found {len(image_records)} images | gpu_ids={gpu_ids} | batch_size={batch_size}",
        flush=True,
    )

    script_path = os.path.abspath(__file__)
    total_start = time.perf_counter()

    procs: List[Tuple[int, int, subprocess.Popen]] = []
    for worker_index, physical_gpu_id, shard in active:
        proc = launch_worker_subprocess(
            script_path=script_path,
            worker_index=worker_index,
            physical_gpu_id=physical_gpu_id,
            image_records=shard,
            output_folder=output_folder,
            batch_size=batch_size,
            attn=attn,
            allow_tf32=allow_tf32,
        )
        procs.append((worker_index, physical_gpu_id, proc))

    failed_workers: List[Dict[str, Any]] = []
    for worker_index, physical_gpu_id, proc in procs:
        return_code = proc.wait()
        print(
            f"[parent] worker={worker_index} physical_gpu={physical_gpu_id} exit_code={return_code}",
            flush=True,
        )
        if return_code != 0:
            failed_workers.append({
                "worker_index": worker_index,
                "physical_gpu_id": physical_gpu_id,
                "exit_code": return_code,
            })

    total_wall_s = time.perf_counter() - total_start
    run_summary = aggregate_final_outputs(
        image_records=image_records,
        output_folder=output_folder,
        gpu_ids_used=[gpu_id for _, gpu_id, _ in active],
        total_wall_s=total_wall_s,
    )

    if failed_workers:
        run_summary["failed_workers"] = failed_workers
        run_summary_path = os.path.join(output_folder, "run_summary.json")
        write_json(run_summary_path, run_summary)

    print(
        f"[parent] Done. total_wall={total_wall_s:.3f}s | "
        f"images={run_summary['num_images_total']} | "
        f"succeeded={run_summary['num_images_succeeded']} | "
        f"failed={run_summary['num_images_failed']}",
        flush=True,
    )


# =========================================================
# CLI
# =========================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-folder", type=str, default=INPUT_FOLDER)
    parser.add_argument("--output-folder", type=str, default=OUTPUT_FOLDER)
    parser.add_argument("--gpu-ids", type=str, default=DEFAULT_GPU_IDS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--attn", type=str, default=DEFAULT_ATTN)
    parser.add_argument("--allow-tf32", action="store_true")

    parser.add_argument("--worker-gpu", type=int, default=None)
    parser.add_argument("--worker-index", type=int, default=None)
    parser.add_argument("--input-list-json", type=str, default=None)
    return parser.parse_args()


# =========================================================
# Main
# =========================================================
if __name__ == "__main__":
    args = parse_args()

    if args.worker_gpu is not None:
        if args.worker_index is None or args.input_list_json is None:
            raise ValueError("Worker mode requires --worker-index and --input-list-json")
        run_worker_mode(
            worker_index=args.worker_index,
            worker_gpu=args.worker_gpu,
            input_list_json=args.input_list_json,
            output_folder=args.output_folder,
            batch_size=max(1, args.batch_size),
            attn=args.attn,
            allow_tf32=args.allow_tf32,
        )
    else:
        run_parent_mode(
            input_folder=args.input_folder,
            output_folder=args.output_folder,
            gpu_ids=parse_gpu_ids(args.gpu_ids),
            batch_size=max(1, args.batch_size),
            attn=args.attn,
            allow_tf32=args.allow_tf32,
        )