import argparse
import csv
import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:
    class _TqdmFallback:
        @staticmethod
        def __call__(iterable, **kwargs):
            return iterable

        @staticmethod
        def write(message):
            print(message)

    tqdm = _TqdmFallback()


NEWAPI_BASE_URL = "https://www.dmxapi.cn"
DEFAULT_MODEL = "gemini-3.5-flash"


DEFAULT_TRAIN_CSV = "/home/xiaolin/Model_36_ArtElingo/arteculture_6792_downsampled_with_country_nonblank_region_split_v2_train.csv"
DEFAULT_OUTPUT_DIR =  "/home/xiaolin/Model_36_ArtElingo/explanation_concept_extraction_gemini35_v2"

CULTURE_TO_FILE = {
    "english": "/home/xiaolin/Model_36_ArtElingo/english_explanations_full.jsonl",
    "chinese": "/home/xiaolin/Model_36_ArtElingo/chinese_explanations_full.jsonl",
    "arabic": "/home/xiaolin/Model_36_ArtElingo/arabic_explanations_full.jsonl",
}

CULTURE_ALIASES = {
    "en": "english",
    "english": "english",
    "zh": "chinese",
    "chinese": "chinese",
    "ar": "arabic",
    "arabic": "arabic",
}

EMOTIONS = {
    "amusement",
    "awe",
    "contentment",
    "excitement",
    "anger",
    "disgust",
    "fear",
    "sadness",
    "something else",
}


def clean_key(key):
    return key.strip().lstrip("\ufeff") if key is not None else key


def normalize_culture(value):
    key = (value or "").strip().lower()
    if key not in CULTURE_ALIASES:
        raise ValueError(f"Unknown culture: {value}. Use one of: english, chinese, arabic, en, zh, ar, all")
    return CULTURE_ALIASES[key]


def parse_cultures(value):
    value = (value or "").strip().lower()
    if value == "all":
        return ["english", "chinese", "arabic"]
    cultures = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        culture = normalize_culture(item)
        if culture not in cultures:
            cultures.append(culture)
    if not cultures:
        raise ValueError("No valid cultures selected.")
    return cultures


def image_basename(path):
    return Path((path or "").replace("\\", "/")).name


def load_train_image_basenames(train_csv):
    train_csv = Path(train_csv)
    if not train_csv.exists():
        raise FileNotFoundError(f"Train CSV not found: {train_csv}")

    basenames = set()
    with train_csv.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        columns = [clean_key(column) for column in (reader.fieldnames or [])]
        if "image_file" not in columns:
            raise KeyError(f"Missing image_file in {train_csv}. Columns: {columns}")
        for row in reader:
            row = {clean_key(k): v for k, v in row.items()}
            basename = image_basename(row.get("image_file"))
            if basename:
                basenames.add(basename)
    return basenames


def normalize_emotion(value):
    text = (value or "").strip().lower()
    return text if text in EMOTIONS else text


def load_explanation_records(culture, explanation_file, train_basenames, min_chars=8):
    explanation_file = Path(explanation_file)
    if not explanation_file.exists():
        raise FileNotFoundError(f"Explanation JSONL not found: {explanation_file}")

    records = []
    image_rows = 0
    kept_image_rows = 0
    with explanation_file.open("r", encoding="utf-8") as file:
        for line_index, line in enumerate(file):
            line = line.strip()
            if not line:
                continue
            image_rows += 1
            obj = json.loads(line)
            basename = obj.get("image_basename") or image_basename(obj.get("image_path"))
            if train_basenames and basename not in train_basenames:
                continue
            kept_image_rows += 1
            explanations = obj.get("explanations") or []
            for exp_index, exp in enumerate(explanations):
                text = (exp.get("text") or "").strip()
                if len(text) < min_chars:
                    continue
                emotion = normalize_emotion(exp.get("emotion") or exp.get("raw_emotion"))
                text_id = f"{culture}_{basename}_{exp_index:03d}"
                records.append(
                    {
                        "text_id": text_id,
                        "culture": culture,
                        "emotion": emotion,
                        "explanation": text,
                        "image_basename": basename,
                        "image_line_index": line_index,
                        "explanation_index": exp_index,
                        "source": exp.get("source", ""),
                        "annotator_culture_background": exp.get("annotator_culture_background", ""),
                        "annotator_native_language": exp.get("annotator_native_language", ""),
                        "annotator_current_country": exp.get("annotator_current_country", ""),
                    }
                )
    return records, {"image_rows": image_rows, "kept_train_image_rows": kept_image_rows}


def chunked(items, batch_size):
    for start in range(0, len(items), batch_size):
        yield start, items[start : start + batch_size]


def make_prompt(culture, batch_records):
    compact_records = [
        {
            "text_id": row["text_id"],
            "culture": row["culture"],
            "emotion": row["emotion"],
            "explanation": row["explanation"],
        }
        for row in batch_records
    ]
    records_json = json.dumps(compact_records, ensure_ascii=False, indent=2)

    return f"""You are an expert annotator for cross-cultural visual art interpretation.

You will receive viewer explanation texts from the {culture} annotation set. Each text is already associated with one emotion label.

Your task is NOT to invent cultural meanings. Your task is to extract concept-association-emotion evidence that is explicitly mentioned or strongly implied by the provided explanation text.

Definitions:
- concept: a concise English noun phrase naming a visual, symbolic, stylistic, scene, color, material, object, person/place, or cultural element mentioned in the explanation.
- concept_original: the same concept phrase in the original explanation language. For English records, this can be the same as concept. For Chinese or Arabic records, use Chinese or Arabic if the phrase is present or can be directly rendered from the text.
- key_association: the meaning, value, memory, symbol, or affective cue linked to that concept, written as a concise English noun phrase.
- key_association_original: the same association phrase in the original explanation language. For English records, this can be the same as key_association. For Chinese or Arabic records, use Chinese or Arabic if supported by the text.
- evidence_span: a short exact span copied from the explanation text that supports the extraction.
- has_cultural_meaning: true only when the explanation connects the concept to a culture-specific, social, ritual, religious, historical, literary, traditional, or symbolic meaning. Use false for general visual mood, beauty, composition, color mood, or personal preference.

Strict rules:
1. Extract concepts only if they are explicitly mentioned or clearly described in the explanation text.
2. Do not infer objects, symbols, or cultural meanings that are not in the text.
3. Do not add external cultural knowledge unless the text itself points to it.
4. If a text gives only a vague feeling without a concrete visual/cultural element, return an empty extractions list for that text.
5. Use concise English noun phrases for concept and key_association.
6. Preserve original-language phrases in concept_original and key_association_original whenever the explanation is Chinese or Arabic. These fields should stay grounded in the explanation, not become free translation.
7. One explanation may contain zero, one, or multiple extractions.
8. Keep evidence_span short and copied from the original text.

Allowed concept_type values:
object, color, scene, style, symbol, material, person_place, action_gesture, composition, other

Allowed association_type values:
cultural_symbolic, ritual_religious, literary_historical, social_custom, visual_affective, personal_memory, uncertain

Allowed confidence values:
high, medium, low

Return ONLY valid JSON in this exact shape:
{{
  "records": [
    {{
      "text_id": "same text_id as input",
      "extractions": [
        {{
          "concept": "concise English noun phrase",
          "concept_original": "concept phrase in the explanation language",
          "concept_type": "object|color|scene|style|symbol|material|person_place|action_gesture|composition|other",
          "key_association": "concise English noun phrase",
          "key_association_original": "association phrase in the explanation language",
          "association_type": "cultural_symbolic|ritual_religious|literary_historical|social_custom|visual_affective|personal_memory|uncertain",
          "has_cultural_meaning": true,
          "evidence_span": "short copied span",
          "confidence": "high|medium|low"
        }}
      ]
    }}
  ]
}}

Input records:
{records_json}
"""


def call_gemini_api(api_key, model, prompt, max_retries=4, timeout=180, max_output_tokens=8192):
    headers = {"Content-Type": "application/json"}
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.0,
            "maxOutputTokens": max_output_tokens,
            "thinkingConfig": {"thinkingBudget": 0},
        },
    }
    url = f"{NEWAPI_BASE_URL}/v1beta/models/{model}:generateContent?key={api_key}"

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
            ).strip()
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


def extract_json_object(text):
    text = (text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        return json.loads(text[start : end + 1])
    raise ValueError("No JSON object found in model response.")


def validate_parsed(parsed, expected_text_ids):
    if not isinstance(parsed, dict):
        raise ValueError("Parsed response is not a JSON object.")
    records = parsed.get("records")
    if not isinstance(records, list):
        raise ValueError("Parsed response missing list field: records.")

    expected = set(expected_text_ids)
    cleaned_records = []
    for item in records:
        if not isinstance(item, dict):
            continue
        text_id = str(item.get("text_id") or "").strip()
        if not text_id or text_id not in expected:
            continue
        extractions = item.get("extractions")
        if not isinstance(extractions, list):
            extractions = []
        cleaned_extractions = []
        for extraction in extractions:
            if not isinstance(extraction, dict):
                continue
            concept = str(extraction.get("concept") or "").strip()
            concept_original = str(extraction.get("concept_original") or concept).strip()
            key_association = str(extraction.get("key_association") or "").strip()
            key_association_original = str(extraction.get("key_association_original") or key_association).strip()
            evidence_span = str(extraction.get("evidence_span") or "").strip()
            if not concept or not key_association or not evidence_span:
                continue
            cleaned_extractions.append(
                {
                    "concept": concept,
                    "concept_original": concept_original,
                    "concept_type": str(extraction.get("concept_type") or "other").strip(),
                    "key_association": key_association,
                    "key_association_original": key_association_original,
                    "association_type": str(extraction.get("association_type") or "uncertain").strip(),
                    "has_cultural_meaning": bool(extraction.get("has_cultural_meaning")),
                    "evidence_span": evidence_span,
                    "confidence": str(extraction.get("confidence") or "medium").strip(),
                }
            )
        cleaned_records.append({"text_id": text_id, "extractions": cleaned_extractions})

    seen = {row["text_id"] for row in cleaned_records}
    for missing_text_id in expected_text_ids:
        if missing_text_id not in seen:
            cleaned_records.append({"text_id": missing_text_id, "extractions": []})
    return {"records": cleaned_records}


def load_done_batch_ids(path):
    done = set()
    path = Path(path)
    if not path.exists():
        return done
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            batch_id = obj.get("batch_id")
            if batch_id:
                done.add(batch_id)
    return done


def append_jsonl(path, obj):
    with Path(path).open("a", encoding="utf-8") as file:
        file.write(json.dumps(obj, ensure_ascii=False) + "\n")


def build_flat_records(batch_result, batch_records):
    by_text_id = {row["text_id"]: row for row in batch_records}
    flat = []
    for parsed_record in batch_result.get("records", []):
        text_id = parsed_record.get("text_id")
        source = by_text_id.get(text_id, {})
        for extraction_index, extraction in enumerate(parsed_record.get("extractions") or []):
            flat.append(
                {
                    "text_id": text_id,
                    "culture": source.get("culture"),
                    "emotion": source.get("emotion"),
                    "image_basename": source.get("image_basename"),
                    "explanation": source.get("explanation"),
                    "extraction_index": extraction_index,
                    **extraction,
                }
            )
    return flat


def write_summary(path, summary):
    with Path(path).open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)


def get_args():
    parser = argparse.ArgumentParser(
        description=(
            "Use Gemini 3.5 to extract concept-culture-key association-emotion tuples "
            "from ArtECulture training explanation texts."
        )
    )
    parser.add_argument(
        "--culture",
        default="all",
        help="Culture(s) to run: english, chinese, arabic, en, zh, ar, comma-separated, or all.",
    )
    parser.add_argument("--train_csv", default=str(DEFAULT_TRAIN_CSV))
    parser.add_argument("--english_jsonl", default=str(CULTURE_TO_FILE["english"]))
    parser.add_argument("--chinese_jsonl", default=str(CULTURE_TO_FILE["chinese"]))
    parser.add_argument("--arabic_jsonl", default=str(CULTURE_TO_FILE["arabic"]))
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument(
        "--api_key",
        default=os.environ.get("NEWAPI_API_KEY", "sk-wGOeW1GNPpb3nWMP2yoXbz4wktH0DM3cVwr85Ab5vyl4TKxj"),
        help="DMXAPI key. Can also be set with NEWAPI_API_KEY.",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--batch_size", type=int, default=25)
    parser.add_argument("--max_batches", type=int, default=None, help="Optional smoke-test limit per culture.")
    parser.add_argument("--min_chars", type=int, default=8)
    parser.add_argument("--max_retries", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--sleep_interval", type=float, default=0.05)
    parser.add_argument("--max_output_tokens", type=int, default=8192)
    parser.add_argument(
        "--retry_failed",
        action="store_true",
        help="Retry batches that appear in the failed JSONL. Successful batches are still skipped.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Load data and print the first prompt without calling the API or writing outputs.",
    )
    return parser.parse_args()


def run_culture(culture, explanation_file, train_basenames, args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    batch_output = output_dir / f"gemini35_explanation_concept_extractions_{culture}.jsonl"
    flat_output = output_dir / f"gemini35_explanation_concept_extractions_{culture}_flat.jsonl"
    failed_output = output_dir / f"gemini35_explanation_concept_extractions_{culture}_failed.jsonl"
    summary_output = output_dir / f"gemini35_explanation_concept_extractions_{culture}_summary.json"

    records, load_stats = load_explanation_records(
        culture,
        explanation_file,
        train_basenames,
        min_chars=args.min_chars,
    )

    batches = []
    for batch_index, (start, batch_records) in enumerate(chunked(records, args.batch_size)):
        if args.max_batches is not None and batch_index >= args.max_batches:
            break
        batch_id = f"{culture}_{batch_index:06d}"
        batches.append((batch_id, start, batch_records))

    done = load_done_batch_ids(batch_output)
    failed_done = set() if args.retry_failed else load_done_batch_ids(failed_output)

    print(f"\n{'=' * 70}")
    print(f"Culture: {culture}")
    print(f"Explanation file: {explanation_file}")
    print(f"Training image rows kept: {load_stats['kept_train_image_rows']} / {load_stats['image_rows']}")
    print(f"Explanation texts: {len(records)}")
    print(f"Batches to consider: {len(batches)} (batch_size={args.batch_size})")
    print(f"Already successful batches skipped: {len(done)}")
    if failed_done:
        print(f"Already failed batches skipped: {len(failed_done)}")
    print(f"{'=' * 70}")

    if args.dry_run:
        if batches:
            prompt = make_prompt(culture, batches[0][2])
            print(prompt[:5000])
        return {
            "culture": culture,
            "records_loaded": len(records),
            "batches": len(batches),
            "dry_run": True,
        }

    processed = 0
    failed = 0
    extracted = 0
    empty_texts = 0

    for batch_id, start, batch_records in tqdm(batches, desc=f"{culture} batches"):
        if batch_id in done or batch_id in failed_done:
            continue

        expected_text_ids = [row["text_id"] for row in batch_records]
        prompt = make_prompt(culture, batch_records)
        raw_text, error = call_gemini_api(
            args.api_key,
            args.model,
            prompt,
            max_retries=args.max_retries,
            timeout=args.timeout,
            max_output_tokens=args.max_output_tokens,
        )

        if raw_text is None:
            failed += 1
            append_jsonl(
                failed_output,
                {
                    "batch_id": batch_id,
                    "culture": culture,
                    "start_index": start,
                    "input_text_ids": expected_text_ids,
                    "error": error or "unknown_error",
                },
            )
            continue

        try:
            parsed = extract_json_object(raw_text)
            parsed = validate_parsed(parsed, expected_text_ids)
        except Exception as parse_error:
            failed += 1
            append_jsonl(
                failed_output,
                {
                    "batch_id": batch_id,
                    "culture": culture,
                    "start_index": start,
                    "input_text_ids": expected_text_ids,
                    "error": f"parse_failed: {parse_error}",
                    "raw_response": raw_text,
                },
            )
            continue

        batch_extractions = sum(len(row.get("extractions") or []) for row in parsed["records"])
        batch_empty_texts = sum(1 for row in parsed["records"] if not row.get("extractions"))
        extracted += batch_extractions
        empty_texts += batch_empty_texts
        processed += 1

        append_jsonl(
            batch_output,
            {
                "batch_id": batch_id,
                "culture": culture,
                "model": args.model,
                "start_index": start,
                "input_text_ids": expected_text_ids,
                "parsed": parsed,
                "raw_response": raw_text,
            },
        )
        for flat_record in build_flat_records(parsed, batch_records):
            append_jsonl(flat_output, flat_record)

        if args.sleep_interval > 0:
            time.sleep(args.sleep_interval)

    summary = {
        "culture": culture,
        "model": args.model,
        "source_explanation_file": str(explanation_file),
        "train_csv": str(args.train_csv),
        "batch_output": str(batch_output),
        "flat_output": str(flat_output),
        "failed_output": str(failed_output),
        "records_loaded": len(records),
        "batches_total": len(batches),
        "batches_processed_this_run": processed,
        "batches_failed_this_run": failed,
        "extractions_written_this_run": extracted,
        "empty_texts_this_run": empty_texts,
        **load_stats,
    }
    write_summary(summary_output, summary)
    return summary


def main():
    args = get_args()
    cultures = parse_cultures(args.culture)

    train_basenames = load_train_image_basenames(args.train_csv)
    explanation_files = {
        "english": Path(args.english_jsonl),
        "chinese": Path(args.chinese_jsonl),
        "arabic": Path(args.arabic_jsonl),
    }

    summaries = []
    for culture in cultures:
        summaries.append(run_culture(culture, explanation_files[culture], train_basenames, args))

    if args.dry_run:
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    combined_summary = output_dir / "gemini35_explanation_concept_extractions_run_summary.json"
    write_summary(
        combined_summary,
        {
            "model": args.model,
            "cultures": cultures,
            "train_csv": str(args.train_csv),
            "output_dir": str(output_dir),
            "summaries": summaries,
        },
    )
    print(f"\nWrote summary: {combined_summary}")


if __name__ == "__main__":
    main()
