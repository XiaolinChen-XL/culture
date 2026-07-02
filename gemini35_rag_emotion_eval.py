import argparse
import base64
import csv
import io
import json
import os
import re
import time
import unicodedata
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path


try:
    from PIL import Image
except ImportError:
    Image = None

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable


NEWAPI_BASE_URL = "https://www.dmxapi.cn"
DEFAULT_MODEL = "gemini-3.5-flash"

# SCRIPT_DIR = Path(__file__).resolve().parent
# ARTECULTURE_DIR = SCRIPT_DIR.parents[1]
CONCEPT_DIR =  "/home/xiaolin/Model_37_ArtElingo/Concept/"
LLM_ANALYSIS_DIR = "LLM-analysis/"

DEFAULT_TEST_CSV = "arteculture_6792_downsampled_with_country_nonblank_region_split_v2_test.csv"
DEFAULT_CONCEPT_JSONL = "/home/xiaolin/Model_37_ArtElingo/Concept/step3_image_concepts.jsonl"
DEFAULT_KB_DIR = LLM_ANALYSIS_DIR
DEFAULT_OUTPUT_DIR = CONCEPT_DIR / "RAG-Evaluation" / "outputs"

OLD_IMAGE_PREFIX = "/home/xiaolin/dataset/ArtECulture/Image/"
DEFAULT_IMAGE_BASE = ARTECULTURE_DIR / "Image"

CULTURES = ["english", "chinese", "arabic"]
CULTURE_TO_SHORT = {"english": "en", "chinese": "zh", "arabic": "ar"}
SHORT_TO_CULTURE = {"en": "english", "zh": "chinese", "ar": "arabic"}
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

CATEGORY_WEIGHTS = {
    "symbolic_cultural_motifs": 5,
    "objects_subjects": 4,
    "scenes_settings": 3,
    "compositional_style": 2,
    "colors_color_schemes": 1,
}


def clean_key(key):
    return key.strip().lstrip("\ufeff") if key is not None else key


def image_basename(path):
    return Path((path or "").replace("\\", "/")).name


def normalize_path(path):
    return unicodedata.normalize("NFC", str(path))


def resolve_image_path(image_file, old_prefix, image_base):
    image_file = normalize_path(image_file)
    candidates = []
    if image_file:
        candidates.append(image_file)
    if old_prefix and image_base and image_file.startswith(old_prefix):
        candidates.append(str(Path(image_base) / image_file[len(old_prefix) :]))
    if image_base:
        candidates.append(str(Path(image_base) / image_basename(image_file)))

    for candidate in candidates:
        candidate = normalize_path(candidate)
        if os.path.exists(candidate):
            return candidate

    # Try Unicode-normalized basename matching inside image_base.
    if image_base and os.path.isdir(image_base):
        target_name = normalize_path(image_basename(image_file))
        for filename in os.listdir(image_base):
            if normalize_path(filename) == target_name:
                return str(Path(image_base) / filename)
    return None


def get_image_media_type(file_path):
    ext = Path(file_path).suffix.lower()
    return {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".webp": "image/webp",
    }.get(ext, "image/jpeg")


def encode_image_to_base64_uri(file_path, max_side=2048, jpeg_quality=90):
    if max_side is None or max_side <= 0 or Image is None:
        mime_type = get_image_media_type(file_path)
        with open(file_path, "rb") as file:
            encoded = base64.b64encode(file.read()).decode("utf-8")
        return f"data:{mime_type};base64,{encoded}"

    with Image.open(file_path) as probe:
        width, height = probe.size
        image_format = (probe.format or "").upper()
        mode = probe.mode

    needs_resize = max(width, height) > max_side
    needs_convert = mode not in ("RGB", "L")
    is_supported_format = image_format in ("JPEG", "PNG", "WEBP")

    if not needs_resize and not needs_convert and is_supported_format:
        mime_type = get_image_media_type(file_path)
        with open(file_path, "rb") as file:
            encoded = base64.b64encode(file.read()).decode("utf-8")
        return f"data:{mime_type};base64,{encoded}"

    with Image.open(file_path) as image:
        image.load()
        if needs_convert:
            image = image.convert("RGB")
        if needs_resize:
            scale = max_side / float(max(width, height))
            new_size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
            image = image.resize(new_size, Image.BICUBIC)
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=jpeg_quality)
        encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded}"


def normalize_concept(text):
    text = (text or "").lower().strip()
    text = text.replace("_", " ").replace("-", " ")
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\b(chinese|arabic|english|western|traditional|ancient|old)\b", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    words = []
    for word in text.split():
        if len(word) > 3 and word.endswith("ies"):
            word = word[:-3] + "y"
        elif len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
            word = word[:-1]
        words.append(word)
    return " ".join(words)


def concept_variants(text):
    norm = normalize_concept(text)
    variants = {norm}
    replacements = [
        ("colour", "color"),
        ("color", "colour"),
        ("seal stamp", "seal"),
        ("seal impression", "seal"),
        ("artist seal", "seal"),
        ("red seal", "seal"),
        ("calligraphy character", "calligraphy"),
        ("ink and wash", "ink wash"),
        ("ink and color", "ink wash"),
    ]
    for src, dst in replacements:
        if src in norm:
            variants.add(norm.replace(src, dst))
    return {v for v in variants if v}


def load_test_rows(path):
    rows = []
    with Path(path).open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            rows.append({clean_key(k): (v.strip() if isinstance(v, str) else v) for k, v in row.items()})
    return rows


def load_image_concepts(path):
    by_basename = {}
    with Path(path).open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            obj = json.loads(line)
            basename = image_basename(obj.get("image_file") or obj.get("resolved_image_path"))
            if basename:
                by_basename[basename] = obj
    return by_basename


def flatten_concepts(concept_record):
    items = []
    if not concept_record:
        return items
    for category in CATEGORY_WEIGHTS:
        for concept in concept_record.get(category) or []:
            concept = str(concept).strip()
            if concept:
                items.append(
                    {
                        "concept": concept,
                        "category": category,
                        "category_weight": CATEGORY_WEIGHTS[category],
                    }
                )
    return items


def load_kb_records(kb_dir):
    kb_dir = Path(kb_dir)
    candidate_files = [
        kb_dir / "concept_culture_association_emotion_distribution_tuples_gemini35_english_en.json",
        kb_dir / "concept_culture_association_emotion_distribution_tuples_gemini35_english_zh.json",
        kb_dir / "concept_culture_association_emotion_distribution_tuples_gemini35_english_ar.json",
    ]
    if not all(path.exists() for path in candidate_files):
        candidate_files = [
            kb_dir / "6-extract-tuple" / "concept_culture_association_emotion_distribution_tuples_gemini35_english_en.json",
            kb_dir / "6-extract-tuple" / "concept_culture_association_emotion_distribution_tuples_gemini35_english_zh.json",
            kb_dir / "6-extract-tuple" / "concept_culture_association_emotion_distribution_tuples_gemini35_english_ar.json",
        ]

    records = []
    for path in candidate_files:
        if not path.exists():
            continue
        obj = json.loads(path.read_text(encoding="utf-8"))
        tuples = obj.get("tuples") if isinstance(obj, dict) else obj
        if isinstance(tuples, list):
            records.extend(tuples)
    if not records:
        raise FileNotFoundError(f"No KB tuple records found under: {kb_dir}")
    return records


def build_kb_index(kb_records):
    index = defaultdict(list)
    for record in kb_records:
        culture = record.get("culture")
        concept = record.get("concept", "")
        for variant in concept_variants(concept):
            index[(culture, variant)].append(record)
    return index


def retrieve_kb(concepts, target_culture, kb_index, top_k=6, include_visual_affective=True):
    culture_short = CULTURE_TO_SHORT[target_culture]
    scored = []
    seen = set()
    for item in concepts:
        query = item["concept"]
        for variant in concept_variants(query):
            for record in kb_index.get((culture_short, variant), []):
                if not include_visual_affective and not record.get("has_cultural_meaning"):
                    continue
                key = (record.get("culture"), record.get("concept"))
                if key in seen:
                    continue
                seen.add(key)
                score = item["category_weight"]
                if normalize_concept(query) == normalize_concept(record.get("concept")):
                    score += 2
                if record.get("has_cultural_meaning"):
                    score += 1
                scored.append(
                    {
                        "score": score,
                        "matched_query": query,
                        "matched_category": item["category"],
                        "record": record,
                    }
                )
    scored.sort(key=lambda x: (-x["score"], x["record"].get("concept", "")))
    return scored[:top_k]


def format_emotion_distribution(dist, max_items=4):
    if not isinstance(dist, dict):
        return ""
    items = sorted(dist.items(), key=lambda x: x[1], reverse=True)[:max_items]
    return ", ".join(f"{emotion} {prob:.2f}" for emotion, prob in items)


def build_prompt(mode, target_culture, concept_record, retrieved_entries):
    culture_name = {
        "english": "English-speaking / Western",
        "chinese": "Chinese",
        "arabic": "Arabic-speaking",
    }[target_culture]

    prompt = (
        f"You are interpreting emotions from a {culture_name} cultural perspective.\n\n"
        "Look at this painting and identify the single emotion it evokes for viewers from this target culture.\n"
        "Choose exactly ONE from: amusement, awe, contentment, excitement, anger, disgust, fear, sadness, something else.\n"
    )

    flat = flatten_concepts(concept_record)
    if mode in ("concept", "rag") and flat:
        by_category = defaultdict(list)
        for item in flat:
            by_category[item["category"]].append(item["concept"])
        prompt += "\nDetected visual concepts in the image:\n"
        for category in CATEGORY_WEIGHTS:
            values = by_category.get(category) or []
            if values:
                prompt += f"- {category}: {', '.join(values[:12])}\n"

    if mode == "rag" and retrieved_entries:
        prompt += f"\nRelevant cultural-emotional knowledge for {culture_name} viewers:\n"
        for idx, entry in enumerate(retrieved_entries, start=1):
            record = entry["record"]
            associations = "; ".join((record.get("key_associations") or [])[:5])
            dist = format_emotion_distribution(record.get("emotion_distribution") or {})
            cultural_flag = "yes" if record.get("has_cultural_meaning") else "no"
            prompt += (
                f"{idx}. concept: {record.get('concept')}\n"
                f"   matched image concept: {entry['matched_query']} ({entry['matched_category']})\n"
                f"   key associations: {associations}\n"
                f"   emotion distribution: {dist}\n"
                f"   has cultural meaning: {cultural_flag}\n"
            )
        prompt += (
            "\nUse this knowledge only when it is visually relevant to the image. "
            "Do not force an emotion only because a concept appears in the knowledge base.\n"
        )

    prompt += (
        "\nAlso give a brief reason based on the visual content and, when provided, the relevant cultural knowledge.\n"
        "Respond in exactly this format:\n"
        "emotion: <one emotion label>\n"
        "reason: <brief reason>"
    )
    return prompt


def call_gemini_api(api_key, model, image_path, prompt, max_retries=4, timeout=180, max_image_side=2048, jpeg_quality=90):
    try:
        image_data_uri = encode_image_to_base64_uri(image_path, max_side=max_image_side, jpeg_quality=jpeg_quality)
        data_header, image_data = image_data_uri.split(",", 1)
        media_type = data_header.removeprefix("data:").split(";", 1)[0]
    except Exception as error:
        return None, f"encode_failed: {error}"

    payload = {
        "contents": [
            {
                "parts": [
                    {"inline_data": {"mime_type": media_type, "data": image_data}},
                    {"text": prompt},
                ]
            }
        ],
        "generationConfig": {
            "temperature": 0.0,
            "maxOutputTokens": 768,
            "thinkingConfig": {"thinkingBudget": 0},
        },
    }
    url = f"{NEWAPI_BASE_URL}/v1beta/models/{model}:generateContent?key={api_key}"
    headers = {"Content-Type": "application/json"}

    for attempt in range(max_retries):
        try:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            request = urllib.request.Request(url, data=body, headers=headers, method="POST")
            try:
                with urllib.request.urlopen(request, timeout=timeout) as response:
                    response_body = response.read().decode("utf-8")
                    status_code = response.getcode()
            except urllib.error.HTTPError as http_error:
                detail = http_error.read().decode("utf-8", errors="replace")[:300]
                raise RuntimeError(f"HTTP {http_error.code}: {detail}") from http_error
            if status_code != 200:
                raise RuntimeError(f"HTTP {status_code}: {response_body[:300]}")

            result = json.loads(response_body)
            candidates = result.get("candidates", [])
            if not candidates:
                raise RuntimeError(f"no_candidates: {result.get('promptFeedback') or result}")
            parts = (candidates[0].get("content") or {}).get("parts") or []
            raw_text = "".join(part.get("text", "") for part in parts if isinstance(part, dict)).strip()
            if not raw_text:
                raise RuntimeError(f"empty response: {str(result)[:300]}")
            return raw_text, None
        except Exception as error:
            wait_time = 2 ** attempt
            if attempt < max_retries - 1:
                print(f"  [Retry {attempt + 1}/{max_retries}] {error}; waiting {wait_time}s")
                time.sleep(wait_time)
            else:
                return None, str(error)
    return None, "max_retries_exceeded"


def parse_emotion(text):
    cleaned = (text or "").strip().strip('"').strip("'").strip(".").strip(",")
    lowered = cleaned.lower()
    for line in lowered.splitlines():
        if line.startswith("emotion:"):
            lowered = line.split(":", 1)[1].strip()
            break
    for emotion in sorted(EMOTIONS, key=len, reverse=True):
        if lowered == emotion or re.search(rf"\b{re.escape(emotion)}\b", lowered):
            return emotion
    if any(token in lowered for token in ["other", "something"]):
        return "something else"
    return "UNKNOWN"


def parse_reason(text):
    lines = [line.strip() for line in (text or "").splitlines() if line.strip()]
    for line in lines:
        if line.lower().startswith("reason:"):
            return line.split(":", 1)[1].strip()
    return (text or "").strip()


def load_existing(output_csv, retry_failed=False):
    processed = set()
    if not Path(output_csv).exists() or Path(output_csv).stat().st_size == 0:
        return processed
    with Path(output_csv).open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            if retry_failed and row.get("pred_emotion") == "API_FAILED":
                continue
            processed.add((row.get("image_file"), row.get("target_culture"), row.get("mode")))
    return processed


def compute_metrics(rows):
    valid = [r for r in rows if r.get("pred_emotion") not in ("API_FAILED", "UNKNOWN", "")]
    total = len(valid)
    correct = sum(1 for r in valid if r.get("correct") == "1")
    per_culture = {}
    for culture in CULTURES:
        subset = [r for r in valid if r.get("target_culture") == culture]
        if subset:
            per_culture[culture] = {
                "n": len(subset),
                "accuracy": sum(1 for r in subset if r.get("correct") == "1") / len(subset),
            }
    return {
        "valid_predictions": total,
        "accuracy": correct / total if total else None,
        "per_culture": per_culture,
        "failed_or_unknown": len(rows) - total,
    }


def get_args():
    parser = argparse.ArgumentParser(description="Evaluate Gemini 3.5 with concept/KB RAG prompt augmentation.")
    parser.add_argument("--mode", choices=["baseline", "concept", "rag"], default="rag")
    parser.add_argument("--culture", default="all", help="english, chinese, arabic, or all")
    parser.add_argument("--test_csv", default=str(DEFAULT_TEST_CSV))
    parser.add_argument("--concept_jsonl", default=str(DEFAULT_CONCEPT_JSONL))
    parser.add_argument("--kb_dir", default=str(DEFAULT_KB_DIR))
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--image_base", default=str(DEFAULT_IMAGE_BASE))
    parser.add_argument("--old_image_prefix", default=OLD_IMAGE_PREFIX)
    parser.add_argument("--api_key", default=os.environ.get("NEWAPI_API_KEY", "sk-wGOeW1GNPpb3nWMP2yoXbz4wktH0DM3cVwr85Ab5vyl4TKxj"))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max_samples", type=int, default=None, help="Limit number of test images for smoke tests.")
    parser.add_argument("--top_k", type=int, default=6)
    parser.add_argument("--cultural_only", action="store_true", help="Retrieve only KB entries with has_cultural_meaning=true.")
    parser.add_argument("--max_retries", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--sleep_interval", type=float, default=0.05)
    parser.add_argument("--max_image_side", type=int, default=2048)
    parser.add_argument("--jpeg_quality", type=int, default=90)
    parser.add_argument("--retry_failed", action="store_true")
    parser.add_argument("--dry_run", action="store_true", help="Build inputs and print one prompt without calling Gemini.")
    return parser.parse_args()


def parse_cultures(value):
    value = (value or "").strip().lower()
    if value == "all":
        return CULTURES
    cultures = []
    for item in value.split(","):
        item = item.strip()
        if item in SHORT_TO_CULTURE:
            item = SHORT_TO_CULTURE[item]
        if item not in CULTURES:
            raise ValueError(f"Unknown culture: {item}")
        cultures.append(item)
    return cultures


def main():
    args = get_args()
    cultures = parse_cultures(args.culture)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    test_rows = load_test_rows(args.test_csv)
    if args.max_samples is not None:
        test_rows = test_rows[: args.max_samples]
    concepts_by_image = load_image_concepts(args.concept_jsonl)

    kb_index = {}
    if args.mode == "rag":
        kb_records = load_kb_records(args.kb_dir)
        kb_index = build_kb_index(kb_records)

    output_csv = output_dir / f"gemini35_{args.mode}_test_predictions.csv"
    output_jsonl = output_dir / f"gemini35_{args.mode}_test_predictions.jsonl"
    summary_json = output_dir / f"gemini35_{args.mode}_eval_summary.json"
    prompt_preview = output_dir / f"gemini35_{args.mode}_prompt_preview.txt"

    header = [
        "mode",
        "image_file",
        "image_basename",
        "target_culture",
        "gt_emotion",
        "pred_emotion",
        "correct",
        "pred_reason",
        "country",
        "region_group",
        "concept_count",
        "retrieved_kb_count",
        "status",
        "error",
    ]
    existing = load_existing(output_csv, retry_failed=args.retry_failed)
    file_exists = output_csv.exists() and output_csv.stat().st_size > 0

    predictions = []
    if file_exists:
        with output_csv.open("r", encoding="utf-8-sig", newline="") as file:
            predictions.extend(list(csv.DictReader(file)))

    with output_csv.open("a", encoding="utf-8-sig", newline="") as csv_file, output_jsonl.open("a", encoding="utf-8") as jsonl_file:
        writer = csv.DictWriter(csv_file, fieldnames=header)
        if not file_exists:
            writer.writeheader()

        for row in tqdm(test_rows, desc=f"{args.mode} test images"):
            basename = image_basename(row.get("image_file"))
            concept_record = concepts_by_image.get(basename)
            flat_concepts = flatten_concepts(concept_record)
            image_path = resolve_image_path(row.get("image_file"), args.old_image_prefix, args.image_base)

            for culture in cultures:
                key = (row.get("image_file"), culture, args.mode)
                if key in existing:
                    continue
                gt_emotion = row.get(f"{culture}_emotion", "")
                retrieved = []
                if args.mode == "rag":
                    retrieved = retrieve_kb(
                        flat_concepts,
                        culture,
                        kb_index,
                        top_k=args.top_k,
                        include_visual_affective=not args.cultural_only,
                    )
                prompt = build_prompt(args.mode, culture, concept_record, retrieved)

                if args.dry_run:
                    prompt_preview.write_text(prompt, encoding="utf-8")
                    print(f"Dry run prompt written: {prompt_preview}")
                    print(prompt[:4000])
                    return

                if not image_path:
                    raw_text, error = None, f"image_not_found: {row.get('image_file')}"
                else:
                    raw_text, error = call_gemini_api(
                        args.api_key,
                        args.model,
                        image_path,
                        prompt,
                        max_retries=args.max_retries,
                        timeout=args.timeout,
                        max_image_side=args.max_image_side,
                        jpeg_quality=args.jpeg_quality,
                    )

                if raw_text is None:
                    pred_emotion = "API_FAILED"
                    reason = error or ""
                    status = "API_FAILED"
                else:
                    pred_emotion = parse_emotion(raw_text)
                    reason = parse_reason(raw_text)
                    status = "OK" if pred_emotion != "UNKNOWN" else "UNKNOWN"
                    error = ""

                out = {
                    "mode": args.mode,
                    "image_file": row.get("image_file"),
                    "image_basename": basename,
                    "target_culture": culture,
                    "gt_emotion": gt_emotion,
                    "pred_emotion": pred_emotion,
                    "correct": "1" if pred_emotion == gt_emotion else "0",
                    "pred_reason": reason,
                    "country": row.get("country", ""),
                    "region_group": row.get("region_group", ""),
                    "concept_count": str(len(flat_concepts)),
                    "retrieved_kb_count": str(len(retrieved)),
                    "status": status,
                    "error": error or "",
                }
                writer.writerow(out)
                csv_file.flush()
                predictions.append(out)

                jsonl_file.write(
                    json.dumps(
                        {
                            **out,
                            "detected_concepts": flat_concepts,
                            "retrieved_kb_entries": retrieved,
                            "raw_response": raw_text or "",
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                jsonl_file.flush()
                if args.sleep_interval > 0:
                    time.sleep(args.sleep_interval)

    summary = {
        "mode": args.mode,
        "model": args.model,
        "test_csv": str(args.test_csv),
        "concept_jsonl": str(args.concept_jsonl),
        "kb_dir": str(args.kb_dir),
        "output_csv": str(output_csv),
        "output_jsonl": str(output_jsonl),
        "cultures": cultures,
        "max_samples": args.max_samples,
        "top_k": args.top_k,
        "cultural_only": args.cultural_only,
        "metrics": compute_metrics(predictions),
        "status_counts": dict(Counter(r.get("status") for r in predictions)),
    }
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote predictions: {output_csv}")
    print(f"Wrote details: {output_jsonl}")
    print(f"Wrote summary: {summary_json}")


if __name__ == "__main__":
    main()
