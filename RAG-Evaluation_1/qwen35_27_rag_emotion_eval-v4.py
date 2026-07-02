import argparse
import csv
import difflib
import importlib.util
import json
import os
import re
import time
from collections import Counter
from pathlib import Path


os.environ.setdefault("CUDA_VISIBLE_DEVICES", "3")

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable


SCRIPT_DIR = Path(__file__).resolve().parent
GEMINI_RAG_SCRIPT = SCRIPT_DIR / "gemini35_rag_emotion_eval-v2.py"
DEFAULT_MODEL_PATH = "/home/xiaolin/LLM/Qwen3.5-27B"
PROJECT_DIR = SCRIPT_DIR.parent
DEFAULT_TEST_CSV = "/home/xiaolin/dataset/ArtECulture/arteculture_6792_downsampled_with_country_nonblank_region_split_v2_test.csv"
DEFAULT_KB_DIR = PROJECT_DIR / "LLM-analysis"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "outputs"
DEFAULT_IMAGE_BASE = "/home/xiaolin/dataset/ArtECulture/Image/"
OLD_IMAGE_PREFIX = "/home/xiaolin/dataset/ArtECulture/Image/"
OUTPUT_PREFIX = "qwen35_27"
CONCEPT_CACHE_NAME = f"{OUTPUT_PREFIX}_multilingual_image_concepts.jsonl"
KB_FILENAMES = {
    "english": "concept_culture_association_emotion_distribution_tuples_gemini35_english_en.json",
    "chinese": "concept_culture_association_emotion_distribution_tuples_gemini35_english_zh_with_zh_translations.json",
    "arabic": "concept_culture_association_emotion_distribution_tuples_gemini35_english_ar_with_ar_translations.json",
}
FALLBACK_KB_FILENAMES = {
    "chinese": "concept_culture_association_emotion_distribution_tuples_gemini35_english_zh.json",
    "arabic": "concept_culture_association_emotion_distribution_tuples_gemini35_english_ar.json",
}
CONCEPT_CATEGORIES = [
    "objects_subjects",
    "colors_color_schemes",
    "compositional_style",
    "scenes_settings",
    "symbolic_cultural_motifs",
]

CONCEPT_EXTRACTION_PROMPT = """
Analyze this painting and extract concise visual concepts for later cultural-emotion retrieval.

Return ONLY one valid JSON object with exactly this schema:
{
  "english": {
    "caption": "one sentence English visual caption",
    "objects_subjects": ["short English noun phrase", "..."],
    "colors_color_schemes": ["short English color or palette phrase", "..."],
    "compositional_style": ["short English style/composition phrase", "..."],
    "scenes_settings": ["short English scene or setting phrase", "..."],
    "symbolic_cultural_motifs": ["short English symbolic or cultural motif phrase", "..."]
  },
  "chinese": {
    "caption": "一句中文画面描述",
    "objects_subjects": ["简短中文名词短语", "..."],
    "colors_color_schemes": ["简短中文色彩或配色短语", "..."],
    "compositional_style": ["简短中文风格或构图短语", "..."],
    "scenes_settings": ["简短中文场景或环境短语", "..."],
    "symbolic_cultural_motifs": ["简短中文象征或文化母题短语", "..."]
  },
  "arabic": {
    "caption": "وصف بصري عربي من جملة واحدة",
    "objects_subjects": ["عبارة اسمية عربية قصيرة", "..."],
    "colors_color_schemes": ["عبارة عربية قصيرة عن اللون أو اللوحة", "..."],
    "compositional_style": ["عبارة عربية قصيرة عن الأسلوب أو التكوين", "..."],
    "scenes_settings": ["عبارة عربية قصيرة عن المشهد أو المكان", "..."],
    "symbolic_cultural_motifs": ["عبارة عربية قصيرة عن رمز أو دلالة ثقافية", "..."]
  }
}

Rules:
- Use English only under "english", Chinese only under "chinese", and Arabic only under "arabic".
- Keep each concept short, usually 1-5 words or the equivalent length.
- Include only visible content.
- Do not include emotion labels or interpretations.
- Use empty arrays when a category is not visible.
""".strip()


def load_rag_helpers():
    spec = importlib.util.spec_from_file_location("gemini35_rag_emotion_eval_v2", GEMINI_RAG_SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load helper script: {GEMINI_RAG_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


rag = load_rag_helpers()


def get_args():
    parser = argparse.ArgumentParser(description="Evaluate Qwen3.5-27B with concept/KB RAG prompt augmentation.")
    parser.add_argument("--mode", choices=["baseline", "concept", "rag"], default="rag")
    parser.add_argument("--culture", default="all", help="english, chinese, arabic, or all")
    parser.add_argument("--test_csv", default=str(DEFAULT_TEST_CSV))
    parser.add_argument("--concept_jsonl", default=None, help="Deprecated; Qwen concepts are extracted online and cached.")
    parser.add_argument("--concept_cache_jsonl", default=None, help="Cache for Qwen-extracted image concepts.")
    parser.add_argument("--kb_dir", default=str(DEFAULT_KB_DIR))
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--image_base", default=str(DEFAULT_IMAGE_BASE))
    parser.add_argument("--old_image_prefix", default=OLD_IMAGE_PREFIX)
    parser.add_argument("--model_path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--max_samples", type=int, default=None, help="Limit number of test images for smoke tests.")
    parser.add_argument("--top_k", type=int, default=6)
    parser.add_argument("--cultural_only", action="store_true", help="Retrieve only KB entries with has_cultural_meaning=true.")
    parser.add_argument("--max_new_tokens", type=int, default=192)
    parser.add_argument("--max_concept_tokens", type=int, default=384)
    parser.add_argument("--fuzzy_threshold", type=float, default=0.82)
    parser.add_argument("--separate_culture_calls", action="store_true", help="Run one Qwen generation per culture instead of one combined call for --culture all.")
    parser.add_argument("--sleep_interval", type=float, default=0.0)
    parser.add_argument("--retry_failed", action="store_true")
    parser.add_argument("--dry_run", action="store_true", help="Build inputs and print one prompt without loading Qwen.")
    return parser.parse_args()


def load_qwen_model(model_path):
    import torch
    from accelerate import Accelerator
    from transformers import AutoModelForImageTextToText, AutoProcessor

    print("Loading Qwen3.5-27B model...")
    device = Accelerator().device
    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        dtype=torch.float16,
    ).to(device)
    model.eval()

    processor = AutoProcessor.from_pretrained(
        model_path,
        trust_remote_code=True,
    )
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id

    print(f"Qwen model loaded: {model_path} (device={device})")
    return model, processor, device, torch


def call_qwen_model(model, processor, device, torch_module, image_path, prompt, max_new_tokens=256):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    try:
        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            enable_thinking=False,
        ).to(device)

        with torch_module.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )

        input_len = len(inputs.input_ids[0])
        output_text = processor.batch_decode(
            generated_ids[:, input_len:],
            skip_special_tokens=True,
        )[0].strip()
        if not output_text:
            return None, "empty response"
        return output_text, None
    except Exception as error:
        return None, str(error)


def parse_json_object(text):
    cleaned = (text or "").strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`").strip()
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        return json.loads(cleaned[start : end + 1])


def normalize_single_culture_concepts(obj):
    record = {"caption": str(obj.get("caption") or "").strip()}
    for category in CONCEPT_CATEGORIES:
        values = obj.get(category) or []
        if not isinstance(values, list):
            values = [values]
        record[category] = [str(value).strip() for value in values if str(value).strip()]
    return record


def normalize_concept_bundle(obj, image_file, image_path, raw_response):
    record = {
        "image_file": image_file,
        "resolved_image_path": image_path,
        "raw_response": raw_response,
        "status": "OK",
        "error": "",
        "concepts": {},
    }
    for culture in rag.CULTURES:
        culture_obj = obj.get(culture) or {}
        if not isinstance(culture_obj, dict):
            culture_obj = {}
        record["concepts"][culture] = normalize_single_culture_concepts(culture_obj)
    return record


def extract_concepts_with_qwen(model, processor, device, torch_module, image_file, image_path, max_new_tokens):
    raw_text, error = call_qwen_model(
        model,
        processor,
        device,
        torch_module,
        image_path,
        CONCEPT_EXTRACTION_PROMPT,
        max_new_tokens=max_new_tokens,
    )
    if raw_text is None:
        return None, f"concept_extract_failed: {error}"
    try:
        obj = parse_json_object(raw_text)
        return normalize_concept_bundle(obj, image_file, image_path, raw_text), None
    except Exception as parse_error:
        return None, f"concept_parse_failed: {parse_error}; raw={raw_text[:500]}"


def load_concept_cache(path):
    by_basename = {}
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return by_basename
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            obj = json.loads(line)
            if obj.get("status") != "OK":
                continue
            if "concepts" not in obj:
                obj = {
                    "image_file": obj.get("image_file"),
                    "resolved_image_path": obj.get("resolved_image_path"),
                    "raw_response": obj.get("raw_response", ""),
                    "status": "OK",
                    "error": "",
                    "concepts": {"english": normalize_single_culture_concepts(obj)},
                }
            basename = rag.image_basename(obj.get("image_file") or obj.get("resolved_image_path"))
            if basename:
                by_basename[basename] = obj
    return by_basename


def append_concept_cache(path, record):
    with Path(path).open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")


def validate_existing_header(output_csv, expected_header):
    if not output_csv.exists() or output_csv.stat().st_size == 0:
        return
    with output_csv.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.reader(file)
        existing_header = next(reader, [])
    if existing_header != expected_header:
        raise ValueError(
            f"Existing output CSV has incompatible header: {output_csv}. "
            "Use a new --output_dir or remove/rename the old output file."
        )


def load_kb_records(kb_dir):
    kb_dir = Path(kb_dir)
    records_by_culture = {}
    for culture, filename in KB_FILENAMES.items():
        path = kb_dir / filename
        if not path.exists() and culture in FALLBACK_KB_FILENAMES:
            path = kb_dir / FALLBACK_KB_FILENAMES[culture]
        if not path.exists():
            raise FileNotFoundError(
                f"Missing KB file for {culture}: {path}\n"
                f"Qwen default kb_dir is: {DEFAULT_KB_DIR}\n"
                "If your KB is elsewhere, pass it explicitly, for example:\n"
                "  --kb_dir /home/xiaolin/Model_37_ArtECulture/LLM-analysis"
            )
        obj = json.loads(path.read_text(encoding="utf-8"))
        tuples = obj.get("tuples") if isinstance(obj, dict) else obj
        if not isinstance(tuples, list) or not tuples:
            raise FileNotFoundError(f"No KB tuple records found in: {path}")
        records_by_culture[culture] = tuples
    return records_by_culture


def concept_field_for_culture(culture):
    return {"english": "concept", "chinese": "concept_zh", "arabic": "concept_ar"}[culture]


def associations_field_for_culture(culture):
    return {"english": "key_associations", "chinese": "key_associations_zh", "arabic": "key_associations_ar"}[culture]


def normalize_local_concept(text, culture):
    text = str(text or "").lower().strip()
    if culture == "english":
        return rag.normalize_concept(text)
    text = re.sub(r"[\s\-_]+", "", text)
    text = re.sub(r"[^\w\u0600-\u06FF\u4E00-\u9FFF]", "", text)
    return text


def local_concept_variants(text, culture):
    norm = normalize_local_concept(text, culture)
    if not norm:
        return set()
    if culture == "english":
        return rag.concept_variants(text)
    return {norm}


def fuzzy_similarity(a, b):
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    if a in b or b in a:
        return 0.92
    return difflib.SequenceMatcher(None, a, b).ratio()


def retrieve_localized_kb(concepts, target_culture, kb_records_by_culture, top_k=6, include_visual_affective=True, fuzzy_threshold=0.82):
    records = kb_records_by_culture.get(target_culture) or []
    concept_field = concept_field_for_culture(target_culture)
    scored = []
    seen = set()
    for item in concepts:
        query = item["concept"]
        query_variants = local_concept_variants(query, target_culture)
        if not query_variants:
            continue
        for record in records:
            if not include_visual_affective and not record.get("has_cultural_meaning"):
                continue
            record_concept = record.get(concept_field) or record.get("concept", "")
            record_norm = normalize_local_concept(record_concept, target_culture)
            if not record_norm:
                continue
            similarity = max(fuzzy_similarity(query_variant, record_norm) for query_variant in query_variants)
            if similarity < fuzzy_threshold:
                continue
            key = (record.get("culture"), record.get("concept"), record.get(concept_field))
            if key in seen:
                continue
            seen.add(key)
            score = item["category_weight"] + similarity * 3
            if similarity >= 0.999:
                score += 2
            if record.get("has_cultural_meaning"):
                score += 1
            scored.append(
                {
                    "score": score,
                    "similarity": similarity,
                    "matched_query": query,
                    "matched_category": item["category"],
                    "matched_kb_concept": record_concept,
                    "record": record,
                }
            )
    scored.sort(key=lambda x: (-x["score"], -x["similarity"], x["record"].get("concept", "")))
    return scored[:top_k]


def culture_display_name(culture):
    return {
        "english": "English-speaking / Western",
        "chinese": "Chinese",
        "arabic": "Arabic-speaking",
    }[culture]


def format_detected_concepts(concept_record, culture=None):
    flat = rag.flatten_concepts(concept_record)
    if not flat:
        return ""
    by_category = {}
    for item in flat:
        by_category.setdefault(item["category"], []).append(item["concept"])
    label = f"Detected visual concepts for {culture_display_name(culture)} viewers" if culture else "Detected visual concepts in the image"
    lines = [f"{label}:"]
    for category in rag.CATEGORY_WEIGHTS:
        values = by_category.get(category) or []
        if values:
            lines.append(f"- {category}: {', '.join(values[:12])}")
    return "\n".join(lines)


def format_retrieved_entries(culture, retrieved_entries):
    if not retrieved_entries:
        return ""
    concept_field = concept_field_for_culture(culture)
    associations_field = associations_field_for_culture(culture)
    lines = [f"Relevant cultural-emotional knowledge for {culture_display_name(culture)} viewers:"]
    for idx, entry in enumerate(retrieved_entries, start=1):
        record = entry["record"]
        associations = "; ".join((record.get(associations_field) or record.get("key_associations") or [])[:5])
        dist = rag.format_emotion_distribution(record.get("emotion_distribution") or {})
        cultural_flag = "yes" if record.get("has_cultural_meaning") else "no"
        lines.extend(
            [
                f"{idx}. concept: {record.get(concept_field) or record.get('concept')}",
                f"   English concept: {record.get('concept')}",
                f"   matched image concept: {entry['matched_query']} ({entry['matched_category']}, similarity {entry.get('similarity', 1.0):.2f})",
                f"   key associations: {associations}",
                f"   emotion distribution: {dist}",
                f"   has cultural meaning: {cultural_flag}",
            ]
        )
    return "\n".join(lines)


def build_multi_culture_prompt(mode, cultures, concept_by_culture, retrieved_by_culture):
    culture_lines = "\n".join(f"- {culture}: {culture_display_name(culture)}" for culture in cultures)
    prompt = (
        "Look at this painting and identify the single emotion it evokes for each target culture below.\n"
        "Choose exactly ONE emotion for each culture from: amusement, awe, contentment, excitement, "
        "anger, disgust, fear, sadness, something else.\n\n"
        f"Target cultures:\n{culture_lines}\n"
    )

    if mode in ("concept", "rag"):
        for culture in cultures:
            concept_text = format_detected_concepts(concept_by_culture.get(culture), culture=culture)
            if concept_text:
                prompt += f"\n{concept_text}\n"

    if mode == "rag":
        for culture in cultures:
            knowledge_text = format_retrieved_entries(culture, retrieved_by_culture.get(culture) or [])
            if knowledge_text:
                prompt += f"\n{knowledge_text}\n"
        prompt += (
            "\nUse the cultural-emotional knowledge only when it is visually relevant to the image. "
            "Do not force an emotion only because a concept appears in the knowledge base.\n"
        )

    culture_json = ",\n  ".join(
        f'"{culture}": {{"emotion": "<one emotion label>", "reason": "<brief reason>"}}'
        for culture in cultures
    )
    prompt += (
        "\nRespond ONLY as one valid JSON object in exactly this schema:\n"
        "{\n  "
        + culture_json
        + "\n}"
    )
    return prompt


def parse_multi_prediction(text, cultures):
    obj = parse_json_object(text)
    parsed = {}
    for culture in cultures:
        value = obj.get(culture) or {}
        if not isinstance(value, dict):
            value = {}
        emotion_text = str(value.get("emotion") or "")
        reason = str(value.get("reason") or "").strip()
        parsed[culture] = {
            "emotion": rag.parse_emotion(emotion_text),
            "reason": reason or text.strip(),
        }
    return parsed


def main():
    args = get_args()
    cultures = rag.parse_cultures(args.culture)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    concept_cache_jsonl = Path(args.concept_cache_jsonl) if args.concept_cache_jsonl else output_dir / CONCEPT_CACHE_NAME

    test_rows = rag.load_test_rows(args.test_csv)
    if args.max_samples is not None:
        test_rows = test_rows[: args.max_samples]
    concepts_by_image = load_concept_cache(concept_cache_jsonl)

    kb_index = {}
    if args.mode == "rag":
        kb_records_by_culture = load_kb_records(args.kb_dir)
    else:
        kb_records_by_culture = {}

    output_csv = output_dir / f"{OUTPUT_PREFIX}_{args.mode}_test_predictions.csv"
    output_jsonl = output_dir / f"{OUTPUT_PREFIX}_{args.mode}_test_predictions.jsonl"
    summary_json = output_dir / f"{OUTPUT_PREFIX}_{args.mode}_eval_summary.json"
    prompt_preview = output_dir / f"{OUTPUT_PREFIX}_{args.mode}_prompt_preview.txt"
    concept_prompt_preview = output_dir / f"{OUTPUT_PREFIX}_concept_prompt_preview.txt"

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
        "concept_status",
        "concept_error",
        "status",
        "error",
    ]

    existing = rag.load_existing(output_csv, retry_failed=args.retry_failed)
    validate_existing_header(output_csv, header)
    file_exists = output_csv.exists() and output_csv.stat().st_size > 0

    predictions = []
    if file_exists:
        with output_csv.open("r", encoding="utf-8-sig", newline="") as file:
            for row in csv.DictReader(file):
                if args.retry_failed and row.get("pred_emotion") == "API_FAILED":
                    continue
                predictions.append(row)

    model = processor = device = torch_module = None
    if not args.dry_run:
        model, processor, device, torch_module = load_qwen_model(args.model_path)

    with output_csv.open("a", encoding="utf-8-sig", newline="") as csv_file, output_jsonl.open("a", encoding="utf-8") as jsonl_file:
        writer = csv.DictWriter(csv_file, fieldnames=header)
        if not file_exists:
            writer.writeheader()

        for row in tqdm(test_rows, desc=f"{OUTPUT_PREFIX} {args.mode} test images"):
            basename = rag.image_basename(row.get("image_file"))
            image_path = rag.resolve_image_path(row.get("image_file"), args.old_image_prefix, args.image_base)
            concept_bundle = None
            concept_error = ""

            if args.mode in ("concept", "rag"):
                concept_bundle = concepts_by_image.get(basename)
                if concept_bundle is None and args.dry_run:
                    concept_prompt_preview.write_text(CONCEPT_EXTRACTION_PROMPT, encoding="utf-8")
                    print(f"Dry run concept prompt written: {concept_prompt_preview}")
                elif concept_bundle is None and not image_path:
                    concept_error = f"image_not_found_for_concept: {row.get('image_file')}"
                elif concept_bundle is None:
                    concept_bundle, concept_error = extract_concepts_with_qwen(
                        model,
                        processor,
                        device,
                        torch_module,
                        row.get("image_file"),
                        image_path,
                        max_new_tokens=args.max_concept_tokens,
                    )
                    if concept_bundle is not None:
                        concepts_by_image[basename] = concept_bundle
                        append_concept_cache(concept_cache_jsonl, concept_bundle)

            concept_by_culture = {}
            flat_concepts_by_culture = {}
            for culture in cultures:
                if concept_bundle and isinstance(concept_bundle.get("concepts"), dict):
                    concept_by_culture[culture] = concept_bundle["concepts"].get(culture) or {}
                else:
                    concept_by_culture[culture] = {}
                flat_concepts_by_culture[culture] = rag.flatten_concepts(concept_by_culture[culture])

            use_combined_call = len(cultures) > 1 and not args.separate_culture_calls
            if use_combined_call:
                pending_cultures = [
                    culture
                    for culture in cultures
                    if (row.get("image_file"), culture, args.mode) not in existing
                ]
                if not pending_cultures:
                    continue

                retrieved_by_culture = {}
                for culture in pending_cultures:
                    if args.mode == "rag":
                        retrieved_by_culture[culture] = retrieve_localized_kb(
                            flat_concepts_by_culture.get(culture) or [],
                            culture,
                            kb_records_by_culture,
                            top_k=args.top_k,
                            include_visual_affective=not args.cultural_only,
                            fuzzy_threshold=args.fuzzy_threshold,
                        )
                    else:
                        retrieved_by_culture[culture] = []

                prompt = build_multi_culture_prompt(args.mode, pending_cultures, concept_by_culture, retrieved_by_culture)

                if args.dry_run:
                    prompt_preview.write_text(prompt, encoding="utf-8")
                    print(f"Dry run prompt written: {prompt_preview}")
                    print(prompt[:4000])
                    return

                if not image_path:
                    raw_text, error = None, f"image_not_found: {row.get('image_file')}"
                elif concept_error:
                    raw_text, error = None, concept_error
                else:
                    raw_text, error = call_qwen_model(
                        model,
                        processor,
                        device,
                        torch_module,
                        image_path,
                        prompt,
                        max_new_tokens=args.max_new_tokens,
                    )

                if raw_text is None:
                    parsed_predictions = {
                        culture: {"emotion": "API_FAILED", "reason": error or ""}
                        for culture in pending_cultures
                    }
                    parse_error = error or ""
                else:
                    try:
                        parsed_predictions = parse_multi_prediction(raw_text, pending_cultures)
                        parse_error = ""
                    except Exception as error_obj:
                        parsed_predictions = {
                            culture: {"emotion": "UNKNOWN", "reason": raw_text}
                            for culture in pending_cultures
                        }
                        parse_error = f"multi_parse_failed: {error_obj}"

                for culture in pending_cultures:
                    gt_emotion = row.get(f"{culture}_emotion", "")
                    pred_emotion = parsed_predictions[culture]["emotion"]
                    reason = parsed_predictions[culture]["reason"]
                    status = "OK" if pred_emotion not in ("API_FAILED", "UNKNOWN") else pred_emotion
                    row_error = parse_error if pred_emotion == "UNKNOWN" else (error or "")
                    retrieved = retrieved_by_culture.get(culture) or []
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
                        "concept_count": str(len(flat_concepts_by_culture.get(culture) or [])),
                        "retrieved_kb_count": str(len(retrieved)),
                        "concept_status": "OK" if concept_bundle else ("FAILED" if concept_error else "SKIPPED"),
                        "concept_error": concept_error,
                        "status": status,
                        "error": row_error,
                    }
                    writer.writerow(out)
                    csv_file.flush()
                    predictions.append(out)
                    jsonl_file.write(
                        json.dumps(
                            {
                                **out,
                                "detected_concepts": flat_concepts_by_culture.get(culture) or [],
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
                continue

            for culture in cultures:
                key = (row.get("image_file"), culture, args.mode)
                if key in existing:
                    continue

                gt_emotion = row.get(f"{culture}_emotion", "")
                retrieved = []
                if args.mode == "rag":
                    retrieved = retrieve_localized_kb(
                        flat_concepts_by_culture.get(culture) or [],
                        culture,
                        kb_records_by_culture,
                        top_k=args.top_k,
                        include_visual_affective=not args.cultural_only,
                        fuzzy_threshold=args.fuzzy_threshold,
                    )
                prompt = build_multi_culture_prompt(args.mode, [culture], concept_by_culture, {culture: retrieved})

                if args.dry_run:
                    prompt_preview.write_text(prompt, encoding="utf-8")
                    print(f"Dry run prompt written: {prompt_preview}")
                    print(prompt[:4000])
                    return

                if not image_path:
                    raw_text, error = None, f"image_not_found: {row.get('image_file')}"
                elif concept_error:
                    raw_text, error = None, concept_error
                else:
                    raw_text, error = call_qwen_model(
                        model,
                        processor,
                        device,
                        torch_module,
                        image_path,
                        prompt,
                        max_new_tokens=args.max_new_tokens,
                    )

                if raw_text is None:
                    pred_emotion = "API_FAILED"
                    reason = error or ""
                    status = "API_FAILED"
                else:
                    try:
                        parsed = parse_multi_prediction(raw_text, [culture])[culture]
                        pred_emotion = parsed["emotion"]
                        reason = parsed["reason"]
                        status = "OK" if pred_emotion != "UNKNOWN" else "UNKNOWN"
                        error = "" if pred_emotion != "UNKNOWN" else "multi_parse_unknown"
                    except Exception as parse_error:
                        pred_emotion = "UNKNOWN"
                        reason = raw_text
                        status = "UNKNOWN"
                        error = f"multi_parse_failed: {parse_error}"

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
                    "concept_count": str(len(flat_concepts_by_culture.get(culture) or [])),
                    "retrieved_kb_count": str(len(retrieved)),
                    "concept_status": "OK" if concept_bundle else ("FAILED" if concept_error else "SKIPPED"),
                    "concept_error": concept_error,
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
                            "detected_concepts": flat_concepts_by_culture.get(culture) or [],
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
        "model": args.model_path,
        "test_csv": str(args.test_csv),
        "concept_jsonl": None,
        "concept_cache_jsonl": str(concept_cache_jsonl),
        "kb_dir": str(args.kb_dir),
        "output_csv": str(output_csv),
        "output_jsonl": str(output_jsonl),
        "cultures": cultures,
        "max_samples": args.max_samples,
        "top_k": args.top_k,
        "cultural_only": args.cultural_only,
        "fuzzy_threshold": args.fuzzy_threshold,
        "metrics": rag.compute_metrics(predictions),
        "status_counts": dict(Counter(r.get("status") for r in predictions)),
    }
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote predictions: {output_csv}")
    print(f"Wrote details: {output_jsonl}")
    print(f"Wrote summary: {summary_json}")


if __name__ == "__main__":
    main()
