import argparse
import base64
import csv
import io
import os
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from collections import Counter

import requests
from PIL import Image
from tqdm import tqdm


NEWAPI_BASE_URL = "https://www.dmxapi.cn"
DEFAULT_MODEL = "gemini-3.5-flash"
IMAGE_BASE = "/home/xiaolin/dataset/ArtECulture/Image/"
OLD_PREFIX = "YOUR/PATH/TO/WIKIART/"
DATA_CSV = "/home/xiaolin/dataset/ArtECulture/arteculture_6792_downsampled_with_country_nonblank.csv"
LOCAL_DATA_CSV = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "arteculture_6792_downsampled_with_country_nonblank.csv",
)

CULTURES = ["english", "chinese", "arabic"]
EMOTIONS = [
    "amusement",
    "awe",
    "contentment",
    "excitement",
    "anger",
    "disgust",
    "fear",
    "sadness",
    "something else",
]

EMOTION_TRANSLATIONS = {
    "english": {
        "amusement": "amusement",
        "awe": "awe",
        "contentment": "contentment",
        "excitement": "excitement",
        "anger": "anger",
        "disgust": "disgust",
        "fear": "fear",
        "sadness": "sadness",
        "something else": "something else",
    },
    "chinese": {
        "amusement": "娱乐",
        "awe": "敬畏",
        "contentment": "满足",
        "excitement": "兴奋",
        "anger": "愤怒",
        "disgust": "厌恶",
        "fear": "恐惧",
        "sadness": "悲伤",
        "something else": "其他",
    },
    "arabic": {
        "amusement": "تسلية",
        "awe": "رهبة",
        "contentment": "رضا",
        "excitement": "حماس",
        "anger": "غضب",
        "disgust": "اشمئزاز",
        "fear": "خوف",
        "sadness": "حزن",
        "something else": "شيء آخر",
    },
}

PROMPTS = {
    "english": (
        "You are interpreting emotions from a Western/English-speaking cultural perspective.\n\n"
        "Look at this painting and identify the single emotion it evokes. "
        "Choose exactly ONE from: amusement, awe, contentment, excitement, "
        "anger, disgust, fear, sadness, something else.\n"
        "Also give a brief reason based on the visual content.\n"
        "Respond in exactly this format:\n"
        "emotion: <one emotion label>\n"
        "reason: <brief reason>"
    ),
    "chinese": (
        "请从中国文化的视角来理解这幅画所表达的情感。\n\n"
        "观察这幅画，判断它最能唤起下列哪一种情感。"
        "请从以下九个选项中选择恰好一个：娱乐、敬畏、满足、兴奋、愤怒、厌恶、恐惧、悲伤、其他。\n"
        "同时根据画面内容给出一个简短理由。\n"
        "请严格按照以下格式回答：\n"
        "emotion: <一个情感标签>\n"
        "reason: <简短理由>"
    ),
    "arabic": (
        "أنت تفسر المشاعر من منظور ثقافي عربي.\n\n"
        "انظر إلى هذه اللوحة وحدد الشعور الواحد الذي تثيره. "
        "اختر شعورا واحدا فقط من القائمة التالية: تسلية، رهبة، رضا، حماس، غضب، اشمئزاز، خوف، حزن، شيء آخر.\n"
        "قدم أيضا سببا موجزا بناء على المحتوى البصري.\n"
        "أجب بهذا التنسيق فقط:\n"
        "emotion: <تسمية شعور واحدة>\n"
        "reason: <سبب موجز>"
    ),
}


# Per-model filename stem used for both per-culture output CSVs and the combined CSV.
OUTPUT_STEM = "gemini35_flash_preview_pred_image_only_arteculture"


def normalize_path(path):
    return unicodedata.normalize("NFC", path)


def resolve_image_path(image_file, old_prefix, image_base):
    actual_path = normalize_path(image_file.replace(old_prefix, image_base))
    if os.path.exists(actual_path):
        return actual_path

    dir_path = os.path.dirname(actual_path)
    target_name = normalize_path(os.path.basename(actual_path))
    if os.path.isdir(dir_path):
        for filename in os.listdir(dir_path):
            if normalize_path(filename) == target_name:
                return os.path.join(dir_path, filename)
    return None


def get_image_media_type(file_path):
    ext = Path(file_path).suffix.lower()
    media_types = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".webp": "image/webp",
    }
    return media_types.get(ext, "image/jpeg")


def encode_image_to_base64_uri(file_path, max_side=2048, jpeg_quality=90):
    """Encode an image to a base64 data URI with three speed optimizations:

    1. Fast-path: if the image is already <= max_side and is JPEG/PNG/WebP, skip
       PIL entirely and base64 the raw file bytes (~10ms vs ~300ms).
    2. BICUBIC instead of LANCZOS for downscaling (faster, virtually no visual
       difference for a vision LLM at these sizes).
    3. JPEG save without optimize=True (optimize does multi-pass encoding and is
       the dominant cost when re-encoding every image).
    """
    if max_side is None or max_side <= 0:
        mime_type = get_image_media_type(file_path)
        with open(file_path, "rb") as file:
            encoded = base64.b64encode(file.read()).decode("utf-8")
        return f"data:{mime_type};base64,{encoded}"

    # Cheap probe: PIL only reads the header here, not pixel data.
    with Image.open(file_path) as probe:
        probe_width, probe_height = probe.size
        probe_format = (probe.format or "").upper()
        probe_mode = probe.mode

    needs_resize = max(probe_width, probe_height) > max_side
    needs_convert = probe_mode not in ("RGB", "L")
    is_supported_format = probe_format in ("JPEG", "PNG", "WEBP")

    # Fast-path: image already small enough, no PIL re-encoding needed.
    if not needs_resize and not needs_convert and is_supported_format:
        mime_type = get_image_media_type(file_path)
        with open(file_path, "rb") as file:
            encoded = base64.b64encode(file.read()).decode("utf-8")
        return f"data:{mime_type};base64,{encoded}"

    # Slow path: decode, resize/convert, re-encode.
    with Image.open(file_path) as image:
        image.load()
        if needs_convert:
            image = image.convert("RGB")

        if needs_resize:
            scale = max_side / float(max(probe_width, probe_height))
            new_size = (
                max(1, int(round(probe_width * scale))),
                max(1, int(round(probe_height * scale))),
            )
            image = image.resize(new_size, Image.BICUBIC)

        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=jpeg_quality)
        encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")

    return f"data:image/jpeg;base64,{encoded}"


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--culture",
        default="chinese,arabic",
        help=(
            "Culture(s) to run. Comma-separated list, e.g. 'chinese,arabic'. "
            "Use 'all' to run english + chinese + arabic. "
            "Valid options: english, chinese, arabic, all."
        ),
    )
    parser.add_argument("--input_csv", default=DATA_CSV)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument(
        "--api_key",
        default=os.environ.get("NEWAPI_API_KEY", "sk-wGOeW1GNPpb3nWMP2yoXbz4wktH0DM3cVwr85Ab5vyl4TKxj"), # sk-xxxxxxxxxx
        help="DMXAPI key. Can also be set with NEWAPI_API_KEY.",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Gemini model name, default: gemini-3.5-flash")
    parser.add_argument("--max_retries", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=180,
                        help="HTTP timeout in seconds. Default raised to 180 to handle slow uploads.")
    parser.add_argument(
        "--sleep_interval",
        type=float,
        default=0.05,
        help="Seconds to sleep after each API call.",
    )
    parser.add_argument(
        "--max_image_side",
        type=int,
        default=2048,
        help="Resize so the longer side <= this many pixels before base64-encoding. "
             "Set 0 to disable (will likely cause write timeouts on large paintings).",
    )
    parser.add_argument(
        "--jpeg_quality",
        type=int,
        default=90,
        help="JPEG quality used when re-encoding resized images. 90 is ~2x faster than 95 with no visible difference.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of concurrent API requests. 8 is a safe default; raise to 16-32 if dmxapi allows. Set 1 to disable concurrency.",
    )
    parser.add_argument(
        "--retry_failed",
        action="store_true",
        help="If set, rows previously marked API_FAILED will be retried instead of skipped. "
             "Note: this will produce duplicate rows in the CSV for those images (append-only); "
             "you'd need to dedupe afterwards.",
    )
    return parser.parse_args()


def resolve_input_csv(csv_path):
    if os.path.exists(csv_path):
        return csv_path
    if os.path.exists(LOCAL_DATA_CSV):
        return LOCAL_DATA_CSV
    raise FileNotFoundError(f"Input CSV not found: {csv_path} or {LOCAL_DATA_CSV}")


def clean_key(key):
    return key.strip().lstrip("\ufeff") if key is not None else key


def load_samples(csv_path):
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        samples = [
            {clean_key(key): (value.strip() if isinstance(value, str) else value) for key, value in row.items()}
            for row in reader
        ]
        columns = [clean_key(column) for column in (reader.fieldnames or [])]

    required_columns = ["image_file"] + [f"{culture}_emotion" for culture in CULTURES]
    missing = [column for column in required_columns if column not in columns]
    if missing:
        raise KeyError(
            f"Missing required columns: {missing}. "
            f"Columns found in {csv_path}: {columns}"
        )
    return samples


def load_existing_predictions(output_csv):
    """Read an existing per-culture output CSV and return a dict
    image_file -> {country, gt_emotion, pred_emotion, pred_reason, correct}.

    Returns an empty dict if the file does not exist or has no data rows.
    """
    if not os.path.exists(output_csv) or os.path.getsize(output_csv) == 0:
        return {}

    processed = {}
    with open(output_csv, "r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            image_file = (row.get("image_file") or "").strip()
            if not image_file:
                continue
            processed[image_file] = {
                "country": (row.get("country") or "").strip(),
                "gt_emotion": (row.get("gt_emotion") or "").strip().lower(),
                "pred_emotion": (row.get("pred_emotion") or "").strip(),
                "pred_reason": (row.get("pred_reason") or ""),
                "correct": (row.get("correct") or "0").strip(),
            }
    return processed


def parse_emotion(text, culture):
    cleaned = text.strip().strip('"').strip("'").strip(".").strip(",").strip()
    cleaned_lower = cleaned.lower()

    translations = EMOTION_TRANSLATIONS[culture]
    for emotion, local_label in translations.items():
        if culture == "english":
            if cleaned_lower == local_label.lower():
                return emotion
        elif cleaned == local_label:
            return emotion

    for emotion, local_label in sorted(translations.items(), key=lambda item: len(item[1]), reverse=True):
        if culture == "english":
            if local_label.lower() in cleaned_lower:
                return emotion
        elif local_label in cleaned:
            return emotion

    for emotion in sorted(EMOTIONS, key=len, reverse=True):
        if cleaned_lower == emotion:
            return emotion

    for emotion in sorted(EMOTIONS, key=len, reverse=True):
        if emotion in cleaned_lower:
            return emotion

    fallback_phrases = ["something", "other", "none", "其他", "其它", "شيء آخر", "غير ذلك"]
    if any(phrase in cleaned or phrase in cleaned_lower for phrase in fallback_phrases):
        return "something else"

    return "UNKNOWN"


def parse_prediction(text, culture):
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    emotion_text = text
    reason = ""

    for line in lines:
        lower_line = line.lower()
        if lower_line.startswith("emotion:"):
            emotion_text = line.split(":", 1)[1].strip()
        elif lower_line.startswith("reason:"):
            reason = line.split(":", 1)[1].strip()

    if not reason:
        for marker in ["reason:", "Reason:", "because", "Because"]:
            if marker in text:
                reason = text.split(marker, 1)[1].strip()
                break
    if not reason:
        reason = text.strip()

    return parse_emotion(emotion_text, culture), reason


def build_prompt(culture):
    return PROMPTS[culture]


def call_gemini_api(api_key, model, image_path, prompt, max_retries=4, timeout=180,
                    max_image_side=2048, jpeg_quality=95):
    try:
        image_data_uri = encode_image_to_base64_uri(
            image_path,
            max_side=max_image_side,
            jpeg_quality=jpeg_quality,
        )
        data_header, image_data = image_data_uri.split(",", 1)
        media_type = data_header.removeprefix("data:").split(";", 1)[0]
    except Exception as error:
        return None, f"encode_failed: {error}"

    headers = {
        "Content-Type": "application/json",
    }
    payload = {
        "contents": [
            {
                "parts": [
                    {
                        "inline_data": {
                            "mime_type": media_type,
                            "data": image_data,
                        }
                    },
                    {"text": prompt},
                ]
            }
        ],
        "generationConfig": {
            "temperature": 0.0,
            "maxOutputTokens": 512,
            "thinkingConfig": {
                "thinkingBudget": 0
            },
        },
    }
    url = f"{NEWAPI_BASE_URL}/v1beta/models/{model}:generateContent?key={api_key}"

    for attempt in range(max_retries):
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=timeout)
            if response.status_code != 200:
                detail = response.text[:200] if response.text else ""
                raise RuntimeError(f"HTTP {response.status_code}: {detail}")

            result = response.json()
            candidates = result.get("candidates", [])
            if not candidates:
                prompt_feedback = result.get("promptFeedback", {})
                raise RuntimeError(f"no_candidates: {prompt_feedback or result}")

            candidate = candidates[0]
            finish_reason = candidate.get("finishReason", "")
            content = candidate.get("content", {}) or {}
            parts = content.get("parts", []) or []
            raw_text = "".join(
                part.get("text", "")
                for part in parts
                if isinstance(part, dict) and isinstance(part.get("text", ""), str)
            )
            raw_text = raw_text.strip()
            if not raw_text:
                raise RuntimeError(f"empty response (finish={finish_reason}): {str(result)[:300]}")
            return raw_text, None
        except Exception as error:
            wait_time = 2 ** attempt
            if attempt < max_retries - 1:
                tqdm.write(f"  [Retry {attempt + 1}/{max_retries}] {error}; waiting {wait_time}s")
                time.sleep(wait_time)
            else:
                return None, str(error)

    return None, "max_retries_exceeded"


def predict_emotion(image_path, culture, args):
    prompt = build_prompt(culture)
    output_text, error = call_gemini_api(
        args.api_key,
        args.model,
        image_path,
        prompt,
        max_retries=args.max_retries,
        timeout=args.timeout,
        max_image_side=args.max_image_side,
        jpeg_quality=args.jpeg_quality,
    )
    if output_text is None:
        return "API_FAILED", error or ""
    return parse_prediction(output_text, culture)


def run_culture(culture, samples, output_dir, predictions_by_image, args):
    print(f"\n{'=' * 60}")
    print(f"Processing culture: {culture} ({len(samples)} samples, workers={args.workers})")
    print(f"{'=' * 60}")

    gt_column = f"{culture}_emotion"
    output_csv = os.path.join(output_dir, f"{OUTPUT_STEM}_{culture}_prompt.csv")
    header = ["country", "image_file", "gt_emotion", "pred_emotion", "pred_reason", "correct"]

    # ---- Resume: load anything already written to disk ----
    existing = load_existing_predictions(output_csv)
    # Decide which existing rows count as "done" (so we skip them this run).
    if args.retry_failed:
        already_done_files = {
            img for img, data in existing.items()
            if data["pred_emotion"] not in ("API_FAILED",)
        }
    else:
        already_done_files = set(existing.keys())

    file_exists = os.path.exists(output_csv) and os.path.getsize(output_csv) > 0
    if existing:
        print(f"  Resuming: {len(existing)} rows in existing CSV, "
              f"{len(already_done_files)} will be skipped "
              f"({'retrying failures' if args.retry_failed else 'incl. failures'}).")

    # ---- Initialize stats from existing rows so the progress bar shows real totals ----
    top1_correct = 0
    total = 0
    unknown = 0
    failed = 0
    emotion_correct = Counter()
    emotion_total = Counter()

    for img_file, data in existing.items():
        pred_emotion = data["pred_emotion"]
        gt_emotion = data["gt_emotion"]
        # Always make existing predictions available for the combined CSV.
        predictions_by_image.setdefault(img_file, {})[culture] = {
            "emotion": pred_emotion,
            "reason": data["pred_reason"],
        }
        # If we're going to retry this row, don't count it in stats — the new row will.
        if img_file not in already_done_files:
            continue
        if pred_emotion == "UNKNOWN":
            unknown += 1
        if pred_emotion == "API_FAILED":
            failed += 1
            continue  # don't count failures toward accuracy denominator
        total += 1
        emotion_total[gt_emotion] += 1
        if data["correct"] == "1":
            top1_correct += 1
            emotion_correct[gt_emotion] += 1

    # ---- Resolve image paths and figure out which images still need API calls ----
    skipped = 0
    entries = []
    for idx, row in enumerate(samples):
        image_file = row["image_file"].strip()
        actual_path = resolve_image_path(image_file, OLD_PREFIX, IMAGE_BASE)
        entries.append((idx, row, actual_path))
        if actual_path is None and image_file not in already_done_files:
            skipped += 1
            if skipped <= 10:
                print(f"  [{idx + 1}] skipped, file not found: {image_file}")

    unique_paths = []
    seen = set()
    for _, row, path in entries:
        image_file = row["image_file"].strip()
        if image_file in already_done_files:
            continue  # don't re-call API for already-done images
        if path is not None and path not in seen:
            seen.add(path)
            unique_paths.append(path)

    print(
        f"  total samples: {len(samples)}, "
        f"already done: {len(already_done_files)}, "
        f"to process (unique images): {len(unique_paths)}, "
        f"skipped (file not found): {skipped}"
    )

    if not unique_paths and len(already_done_files) >= len([e for e in entries if e[2] is not None]):
        print(f"  Nothing new to do for '{culture}'. Existing CSV already covers all resolvable samples.")

    # ---- Open CSV in append mode if it already exists; write header only for fresh file ----
    mode = "a" if file_exists else "w"
    with open(output_csv, mode, encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(header)

        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
            path_to_future = {
                path: pool.submit(predict_emotion, path, culture, args)
                for path in unique_paths
            }

            progress = tqdm(entries, desc=f"[{culture}]", unit="sample", dynamic_ncols=True)
            for idx, row, actual_path in progress:
                image_file = row["image_file"].strip()
                if image_file in already_done_files:
                    # Already in CSV from a previous run — counted in stats above.
                    continue

                gt_emotion = row[gt_column].strip().lower()
                country = row.get("country", "").strip()

                if actual_path is None:
                    continue  # counted in `skipped` above

                pred_emotion, pred_reason = path_to_future[actual_path].result()

                predictions_by_image.setdefault(image_file, {})[culture] = {
                    "emotion": pred_emotion,
                    "reason": pred_reason,
                }

                if pred_emotion == "UNKNOWN":
                    unknown += 1
                elif pred_emotion == "API_FAILED":
                    failed += 1

                correct = 1 if gt_emotion == pred_emotion else 0
                if pred_emotion != "API_FAILED":
                    top1_correct += correct
                    total += 1
                    emotion_total[gt_emotion] += 1
                    if correct:
                        emotion_correct[gt_emotion] += 1

                writer.writerow([country, image_file, gt_emotion, pred_emotion, pred_reason, correct])
                # Flush regularly so a crash doesn't lose all progress.
                if (idx + 1) % 50 == 0:
                    file.flush()

                accuracy = top1_correct / total * 100 if total else 0
                progress.set_postfix({
                    "Top1": f"{accuracy:.2f}%",
                    "done": len(already_done_files) + (total + failed + unknown),
                    "skip": skipped,
                    "unk": unknown,
                    "fail": failed,
                })

    accuracy = top1_correct / total * 100 if total else 0
    print(f"\n{culture} final results")
    print(f"  total: {total}, top-1 correct: {top1_correct}, accuracy: {accuracy:.2f}%")
    print(f"  skipped: {skipped}, unknown: {unknown}, API failed: {failed}")
    print("  per-emotion accuracy:")
    for emotion in EMOTIONS:
        count_total = emotion_total[emotion]
        count_correct = emotion_correct[emotion]
        emotion_accuracy = count_correct / count_total * 100 if count_total else 0
        print(f"    {emotion:16s}: {emotion_accuracy:5.2f}% ({count_correct}/{count_total})")
    print(f"  saved: {output_csv}")

    return {"total": total, "correct": top1_correct, "accuracy": accuracy}


def write_combined_predictions(samples, output_dir, predictions_by_image):
    combined_csv = os.path.join(output_dir, f"{OUTPUT_STEM}_all_culture_prompts.csv")
    header = [
        "country",
        "image_file",
        "gt_english_emotion",
        "pred_english_emotion",
        "pred_english_reason",
        "gt_chinese_emotion",
        "pred_chinese_emotion",
        "pred_chinese_reason",
        "gt_arabic_emotion",
        "pred_arabic_emotion",
        "pred_arabic_reason",
    ]

    with open(combined_csv, "w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(header)
        for row in samples:
            image_file = row["image_file"].strip()
            preds = predictions_by_image.get(image_file, {})
            writer.writerow([
                row.get("country", "").strip(),
                image_file,
                row["english_emotion"].strip().lower(),
                preds.get("english", {}).get("emotion", ""),
                preds.get("english", {}).get("reason", ""),
                row["chinese_emotion"].strip().lower(),
                preds.get("chinese", {}).get("emotion", ""),
                preds.get("chinese", {}).get("reason", ""),
                row["arabic_emotion"].strip().lower(),
                preds.get("arabic", {}).get("emotion", ""),
                preds.get("arabic", {}).get("reason", ""),
            ])
    print(f"\nCombined predictions saved: {combined_csv}")


ARGS = get_args()
input_csv = resolve_input_csv(ARGS.input_csv)
output_dir = ARGS.output_dir or os.getcwd()
os.makedirs(output_dir, exist_ok=True)
samples = load_samples(input_csv)

if ARGS.culture == "all":
    run_cultures = CULTURES
else:
    run_cultures = [c.strip().lower() for c in ARGS.culture.split(",") if c.strip()]
    invalid = [c for c in run_cultures if c not in CULTURES]
    if invalid:
        raise ValueError(f"Invalid culture(s): {invalid}. Valid options: {CULTURES} (or 'all').")
    if not run_cultures:
        raise ValueError("No cultures specified.")

print(f"Input CSV: {input_csv}")
print(f"Output dir: {output_dir}")
print(f"Samples: {len(samples)}")
print(f"Cultures: {', '.join(run_cultures)}")
print(f"Model: {ARGS.model}")
print(f"Max image side: {ARGS.max_image_side if ARGS.max_image_side > 0 else 'no resize'}")
print(f"Timeout: {ARGS.timeout}s")
print(f"Workers: {ARGS.workers}")
print(f"Retry failed rows: {ARGS.retry_failed}")
print(f"API endpoint: {NEWAPI_BASE_URL}/v1beta/models/{ARGS.model}:generateContent")

results_all = {}
predictions_by_image = {}
for culture in run_cultures:
    results_all[culture] = run_culture(culture, samples, output_dir, predictions_by_image, ARGS)

# Combined CSV: write whenever all three per-culture CSVs exist on disk —
# load any cultures we didn't run this time (e.g. english) from their existing file.
per_culture_csvs = {
    c: os.path.join(output_dir, f"{OUTPUT_STEM}_{c}_prompt.csv")
    for c in CULTURES
}
missing_csvs = [c for c, path in per_culture_csvs.items() if not os.path.exists(path)]
if not missing_csvs:
    for culture in CULTURES:
        if culture not in run_cultures:
            existing = load_existing_predictions(per_culture_csvs[culture])
            for img_file, data in existing.items():
                predictions_by_image.setdefault(img_file, {})[culture] = {
                    "emotion": data["pred_emotion"],
                    "reason": data["pred_reason"],
                }
    write_combined_predictions(samples, output_dir, predictions_by_image)
else:
    print(f"\nSkipping combined CSV — missing per-culture files for: {missing_csvs}")

print(f"\n{'=' * 60}")
print("Summary")
print(f"{'=' * 60}")
for culture, result in results_all.items():
    print(f"  {culture:8s}: Top-1 = {result['accuracy']:.2f}% ({result['correct']}/{result['total']})")

total_all = sum(result["total"] for result in results_all.values())
correct_all = sum(result["correct"] for result in results_all.values())
if total_all > 0:
    print(f"  {'overall':8s}: Top-1 = {correct_all / total_all * 100:.2f}% ({correct_all}/{total_all})")