import os
import json
from typing import Any, Dict, List, Optional, Union

import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


# =========================================================
# INTERPOL WP1
# =========================================================
MODEL_NAME = "Qwen/Qwen3-VL-8B-Instruct"
DEVICE_MAP = "auto"
TORCH_DTYPE = "auto"   # or torch.bfloat16
MAX_NEW_TOKENS_DET = 1024
MAX_NEW_TOKENS_OCR = 2048

INPUT_FOLDER = "/home2/tajamul/Qwen3VL/images"
OUTPUT_FOLDER = "./qwen3vl_outputs"

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
    "nude body parts like exposed breasts, genitals, or buttocks, anus, armpits, belly, feet ",
    "school uniform",
    "school logos/badges",
]

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# =========================================================
# Load model + processor
# =========================================================
print("Loading model...")
model = Qwen3VLForConditionalGeneration.from_pretrained(
    MODEL_NAME,
    torch_dtype=TORCH_DTYPE,
    device_map=DEVICE_MAP,
)

processor = AutoProcessor.from_pretrained(MODEL_NAME)
print("Model loaded.")


# =========================================================
# Helpers
# =========================================================
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def is_image_file(filename: str) -> bool:
    return os.path.splitext(filename.lower())[1] in IMAGE_EXTS


def build_messages(image_path: str, prompt: str) -> List[Dict[str, Any]]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": prompt},
            ],
        }
    ]


@torch.inference_mode()
def run_qwen3_vl(
    image_path: str,
    prompt: str,
    max_new_tokens: int,
) -> str:
    messages = build_messages(image_path, prompt)

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )

    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    generated_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
    )

    input_token_len = inputs["input_ids"].shape[1]
    generated_ids = generated_ids[:, input_token_len:]

    output_text = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    return output_text.strip()


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
        gender = item.get("gender", "")
        ethnicity = item.get("ethnicity", "")
        age = item.get("age", "")
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
            "gender": str(gender).strip(),
            "ethnicity": str(ethnicity).strip(),
            "age": str(age).strip(),
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
    return f"""
Locate every visible instance in the image that belongs to these categories:
"{cat_str}"

Rules:
- Return JSON only.
- Use relative coordinates scaled from 0 to 1000.
- If an item is not present, do not include it.
- Allowed labels are only: person, face, tattoo, blood, palm, text, weapon, keyboard, school uniform, school logos/badges, hotel room, nude body parts like exposed breasts, genitals, or buttocks, anus, armpits, belly, feet.
- For text regions, use label "text".
- For weapon-like objects, use label "weapon".
- For visible hotel room scenes or obvious hotel-room context, use label "hotel room".
- When a face/person is detected, infer age, gender, race, ethnicity, nude body parts.

Output format:
[
  {{
    "label": "category_name",
    "bbox_2d": [x1, y1, x2, y2]
  }}
]
""".strip()


def build_person_prompt() -> str:
    return """
Detect all visible persons in the image.

For each detected person, return JSON only in this format:
[
  {
    "label": "person",
    "bbox_2d": [x1, y1, x2, y2],
    "face_bbox_2d": [x1, y1, x2, y2],
    "action": "short visible action if clear, else empty string",
    "gender": "male/female/other/unknown",
    "age": "estimated age or age range if clear, else empty string",
    "ethnicity": "estimated ethnicity/race if clear, else empty string",
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
- when a face is detected, infer age, gender, race, and ethnicity.
- Return JSON only, no markdown.
""".strip()


def build_ocr_prompt() -> str:
    return """
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
""".strip()


# =========================================================
# Drawing
# =========================================================
def flatten_for_drawing(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    draw_items: List[Dict[str, Any]] = []

    # draw general detections, but skip face/person because person pass already handles them
    for item in result.get("general_detections", []):
        label = item.get("label", "")
        if label in {"face", "person"}:
            continue

        draw_items.append({
            "label": label,
            "bbox_2d": item.get("bbox_2d"),
            "text_content": item.get("text_content", ""),
        })

    # draw persons and their face boxes
    for person in result.get("persons", []):
        draw_items.append({
            "label": "person",
            "bbox_2d": person.get("bbox_2d"),
            "text_content": person.get("action", ""),
        })

        face_bbox = person.get("face_bbox_2d")
        if face_bbox:
            face_text_parts = []
            if person.get("gender"):
                face_text_parts.append(person["gender"])
            if person.get("age"):
                face_text_parts.append(person["age"])
            if person.get("ethnicity"):
                face_text_parts.append(person["ethnicity"])


            draw_items.append({
                "label": "face",
                "bbox_2d": face_bbox,
                "text_content": ", ".join(face_text_parts),
            })

    # draw OCR text
    for item in result.get("ocr_texts", []):
        draw_items.append({
            "label": "text",
            "bbox_2d": item.get("bbox_2d"),
            "text_content": item.get("text_content", ""),
        })

    return draw_items


def draw_bboxes(
    image_path: str,
    result: Dict[str, Any],
    output_path: str,
) -> str:
    image = Image.open(image_path).convert("RGB")
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
# Per-image pipeline
# =========================================================
def process_one_image(
    image_path: str,
    output_folder: str,
    categories: List[str],
) -> Dict[str, Any]:
    base_name = os.path.splitext(os.path.basename(image_path))[0]

    print(f"\nProcessing: {image_path}")

    general_prompt = build_general_detection_prompt(categories)
    person_prompt = build_person_prompt()
    ocr_prompt = build_ocr_prompt()

    general_raw = run_qwen3_vl(
        image_path=image_path,
        prompt=general_prompt,
        max_new_tokens=MAX_NEW_TOKENS_DET,
    )
    general_parsed = extract_json_from_text(general_raw)
    general_items = normalize_general_detection_output(general_parsed)

    person_raw = run_qwen3_vl(
        image_path=image_path,
        prompt=person_prompt,
        max_new_tokens=MAX_NEW_TOKENS_DET,
    )
    person_parsed = extract_json_from_text(person_raw)
    persons = normalize_person_output(person_parsed)

    ocr_raw = run_qwen3_vl(
        image_path=image_path,
        prompt=ocr_prompt,
        max_new_tokens=MAX_NEW_TOKENS_OCR,
    )
    ocr_parsed = extract_json_from_text(ocr_raw)
    ocr_items = normalize_ocr_output(ocr_parsed)

    result = {
        "image_path": image_path,
        "categories_requested": categories,
        "general_detections": general_items,
        "persons": persons,
        "ocr_texts": ocr_items,
        "raw_responses": {
            "general_detection": general_raw,
            "person_detection": person_raw,
            "ocr": ocr_raw,
        },
    }

    json_path = os.path.join(output_folder, f"{base_name}.json")
    annotated_path = os.path.join(output_folder, f"{base_name}_annotated.png")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    draw_bboxes(
        image_path=image_path,
        result=result,
        output_path=annotated_path,
    )

    print(f"Saved JSON: {json_path}")
    print(f"Saved annotated image: {annotated_path}")

    return result


def process_folder(
    input_folder: str,
    output_folder: str,
    categories: List[str],
) -> List[Dict[str, Any]]:
    ensure_dir(output_folder)

    image_files = [
        os.path.join(input_folder, f)
        for f in sorted(os.listdir(input_folder))
        if is_image_file(f)
    ]

    if not image_files:
        print(f"No images found in: {input_folder}")
        return []

    all_results = []
    for image_path in image_files:
        try:
            result = process_one_image(
                image_path=image_path,
                output_folder=output_folder,
                categories=categories,
            )
            all_results.append(result)
        except Exception as e:
            print(f"Failed on {image_path}: {e}")

    summary_path = os.path.join(output_folder, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    print(f"\nSaved folder summary: {summary_path}")
    return all_results


# =========================================================
# Main
# =========================================================
if __name__ == "__main__":
    process_folder(
        input_folder=INPUT_FOLDER,
        output_folder=OUTPUT_FOLDER,
        categories=TARGET_CATEGORIES,
    )