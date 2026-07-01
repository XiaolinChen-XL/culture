import argparse
import json
import os
import re
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


NEWAPI_BASE_URL = "https://www.dmxapi.cn"
DEFAULT_MODEL = "gemini-3.5-flash"

SCRIPT_DIR = Path(__file__).resolve().parent
CONCEPT_DIR = SCRIPT_DIR.parent

DEFAULT_INPUT_KB = CONCEPT_DIR / "concept_emotion_kb_train_split_v2_min10_filtered_mklany0p10_meanjs0p10.json"
DEFAULT_BASELINE_KB = CONCEPT_DIR / "concept_emotion_kb_train_split_v2_min10.json"
DEFAULT_OUTPUT_JSONL = SCRIPT_DIR / "concept_emotion_kb_train_split_v2_min10_filtered_mklany0p10_meanjs0p10_cultural_meanings_gemini35_english.jsonl"
DEFAULT_FAILED_JSONL = SCRIPT_DIR / "concept_emotion_kb_train_split_v2_min10_filtered_mklany0p10_meanjs0p10_cultural_meanings_gemini35_english_failed.jsonl"

CULTURE_LABELS = {
    "zh": "Chinese",
    "en": "English-viewing annotators",
    "ar": "Arabic-viewing annotators",
}

MEANING_TYPES = {
    "symbolic",
    "ritual_religious",
    "literary_historical",
    "social_custom",
    "visual_affective",
    "uncertain",
}

CONFIDENCE_LEVELS = {"high", "medium", "low"}


def get_args():
    parser = argparse.ArgumentParser(
        description="Generate cultural-emotional meaning explanations for filtered concept/culture KB entries using Gemini 3.5."
    )
    parser.add_argument("--input_kb", default=str(DEFAULT_INPUT_KB))
    parser.add_argument(
        "--baseline_kb",
        default=str(DEFAULT_BASELINE_KB),
        help="KB used to compute per-culture overall emotion baselines. Default is the unfiltered train KB.",
    )
    parser.add_argument("--output_jsonl", default=str(DEFAULT_OUTPUT_JSONL))
    parser.add_argument("--failed_jsonl", default=str(DEFAULT_FAILED_JSONL))
    parser.add_argument(
        "--retry_source_failed_jsonl",
        default=None,
        help=(
            "When --retry_failed_only is set, read failed record_ids from this file. "
            "If omitted, failed_jsonl is used as both the retry source and the new failure sink."
        ),
    )
    parser.add_argument(
        "--api_key",
        default=os.environ.get("NEWAPI_API_KEY", ""),
        help="DMXAPI/NewAPI key. Prefer setting NEWAPI_API_KEY.",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max_retries", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--sleep_interval", type=float, default=0.05)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max_output_tokens", type=int, default=1024)
    parser.add_argument("--limit", type=int, default=0, help="Process only the first N pending records. 0 means all.")
    parser.add_argument("--culture", default="all", help="all, or comma-separated culture codes: zh,en,ar")
    parser.add_argument(
        "--retry_failed",
        action="store_true",
        help="If set, do not treat records in failed_jsonl as terminal; try them again.",
    )
    parser.add_argument(
        "--retry_failed_only",
        action="store_true",
        help=(
            "Only rerun record_ids found in failed_jsonl and append successful retries "
            "to output_jsonl. Existing successful output rows are still skipped."
        ),
    )
    parser.add_argument(
        "--include_example_texts",
        type=int,
        default=3,
        help="Number of annotation example_texts to include per prompt.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Only build records and print a prompt preview; do not call Gemini and do not require an API key.",
    )
    return parser.parse_args()


def load_kb(path):
    with open(path, "r", encoding="utf-8") as file:
        data = json.load(file)
    if isinstance(data, dict) and "kb" in data:
        return data["kb"]
    if isinstance(data, list):
        return data
    raise ValueError(f"Unsupported KB shape in {path}")


def get_cultures_arg(raw):
    if raw.strip().lower() == "all":
        return {"zh", "en", "ar"}
    cultures = {item.strip().lower() for item in raw.split(",") if item.strip()}
    invalid = cultures - {"zh", "en", "ar"}
    if invalid:
        raise ValueError(f"Invalid culture code(s): {sorted(invalid)}. Use zh,en,ar or all.")
    return cultures


def compute_culture_baselines(kb):
    emotions = sorted({emotion for entry in kb for emotion in entry.get("distribution", {})})
    counts_by_culture = defaultdict(Counter)
    for entry in kb:
        counts_by_culture[entry["culture"]].update(entry.get("emotion_counts", {}))

    baselines = {}
    for culture, counts in counts_by_culture.items():
        total = sum(counts.get(emotion, 0) for emotion in emotions)
        baselines[culture] = {
            emotion: counts.get(emotion, 0) / total if total else 0.0
            for emotion in emotions
        }
    return baselines


def pct(value):
    return f"{value * 100:.1f}%"


def format_distribution(distribution, min_prob=0.0, top_n=None):
    items = sorted(distribution.items(), key=lambda item: item[1], reverse=True)
    if min_prob > 0:
        items = [item for item in items if item[1] >= min_prob]
    if top_n is not None:
        items = items[:top_n]
    return "\n".join(f"  {emotion}: {pct(prob)}" for emotion, prob in items)


def get_dominant_emotions(distribution, threshold=0.05, top_n=5):
    return [
        emotion
        for emotion, prob in sorted(distribution.items(), key=lambda item: item[1], reverse=True)[:top_n]
        if prob >= threshold
    ]


def get_forbidden_emotions(distribution, threshold=0.05):
    return [emotion for emotion, prob in distribution.items() if prob < threshold]


def build_records(kb, baseline_kb, cultures_to_run, include_example_texts):
    baselines = compute_culture_baselines(baseline_kb)
    groups = defaultdict(list)
    for entry in kb:
        groups[(entry["concept"], entry.get("dimension"))].append(entry)

    records = []
    for (concept, dimension), entries in sorted(groups.items()):
        entries_by_culture = {entry["culture"]: entry for entry in entries}
        for culture, entry in sorted(entries_by_culture.items()):
            if culture not in cultures_to_run:
                continue
            record_id = f"{concept}|||{dimension}|||{culture}"
            other_entries = {
                other_culture: other_entry
                for other_culture, other_entry in entries_by_culture.items()
                if other_culture != culture
            }
            records.append(
                {
                    "record_id": record_id,
                    "concept": concept,
                    "dimension": dimension,
                    "culture": culture,
                    "target_entry": entry,
                    "other_entries": other_entries,
                    "culture_baseline": baselines.get(culture, {}),
                    "example_texts": (entry.get("example_texts") or [])[:include_example_texts],
                    "example_image_ids": (entry.get("example_image_ids") or [])[:5],
                }
            )
    return records


def build_prompt(record):
    entry = record["target_entry"]
    culture = record["culture"]
    culture_label = CULTURE_LABELS.get(culture, culture)
    target_distribution = entry.get("distribution", {})
    baseline = record["culture_baseline"]
    dominant = get_dominant_emotions(target_distribution)
    forbidden = get_forbidden_emotions(target_distribution)

    other_blocks = []
    for other_culture, other_entry in sorted(record["other_entries"].items()):
        other_label = CULTURE_LABELS.get(other_culture, other_culture)
        other_blocks.append(
            f"{other_label} ({other_culture}):\n"
            f"{format_distribution(other_entry.get('distribution', {}), min_prob=0.05, top_n=8)}\n"
            f"Support: {other_entry.get('support_annotations', 0)} annotations across "
            f"{other_entry.get('support_images', 0)} images"
        )
    other_text = "\n\n".join(other_blocks) if other_blocks else "No other culture entries are available."

    examples = record["example_texts"]
    if examples:
        examples_text = "\n".join(f"- {text}" for text in examples)
    else:
        examples_text = "- No example annotation texts available."

    dominant_text = ", ".join(dominant) if dominant else "none above 5%"
    forbidden_text = ", ".join(sorted(forbidden)) if forbidden else "none"

    return f"""You are an expert in cross-cultural visual art, symbolism, and emotion annotation.

I will provide statistical evidence from training annotations about how viewers from different cultures emotionally respond to images containing a specific visual concept. Your task is to explain the cultural-emotional meaning for ONE target culture.

Important:
- Use the statistical evidence as the primary grounding.
- You may use cultural knowledge, but do not invent a cultural symbol if the concept is ordinary, visual, or context-dependent.
- First decide whether the concept has a well-established culture-specific symbolic, ritual, literary, historical, or social meaning in the target culture.
- If not, set "has_cultural_meaning": false and explain it as "visual_affective" or "uncertain".
- Do not mention any emotion whose probability is below 5% for the target culture.
- The explanation must be in English.
- The "cultural_meaning" field must be 1-2 English sentences.
- The "cross_cultural_note" field must be exactly one English sentence.
- The "key_associations" field must contain short English phrases.
- Output must be valid JSON only. No markdown, no comments.

[Concept]
concept: "{record['concept']}"
detected in dimension: {record['dimension']}

[Target culture]
Target culture: {culture_label} ({culture})
Emotion distribution for this culture:
{format_distribution(target_distribution, min_prob=0.0, top_n=9)}
Support: {entry.get('support_annotations', 0)} annotations across {entry.get('support_images', 0)} images

Dominant emotions allowed to discuss (>=5%): {dominant_text}
Emotions forbidden to discuss (<5%): {forbidden_text}

[Overall emotion baseline for this culture, not specific to this concept]
{format_distribution(baseline, min_prob=0.0, top_n=9)}

[Cross-cultural comparison for the same concept]
{other_text}

[Training annotation text examples for grounding]
{examples_text}

[Required JSON schema]
{{
  "has_cultural_meaning": true or false,
  "meaning_type": "symbolic | ritual_religious | literary_historical | social_custom | visual_affective | uncertain",
  "interpretation_confidence": "high | medium | low",
  "cultural_meaning": "1-2 English sentences explaining why the target viewers show the dominant emotions, grounded in the statistics and concept.",
  "key_associations": ["specific association 1", "specific association 2"],
  "cross_cultural_note": "one English sentence comparing the target culture with the other cultures, or saying the difference is not clearly symbolic."
}}
"""


def call_gemini_api(api_key, model, prompt, args):
    headers = {"Content-Type": "application/json"}
    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt},
                ]
            }
        ],
        "generationConfig": {
            "temperature": args.temperature,
            "maxOutputTokens": args.max_output_tokens,
            "thinkingConfig": {
                "thinkingBudget": 0,
            },
        },
    }
    url = f"{NEWAPI_BASE_URL}/v1beta/models/{model}:generateContent?key={api_key}"
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")

    for attempt in range(args.max_retries):
        try:
            request = urllib.request.Request(url, data=body, headers=headers, method="POST")
            with urllib.request.urlopen(request, timeout=args.timeout) as response:
                response_body = response.read().decode("utf-8")
            result = json.loads(response_body)
            candidates = result.get("candidates", [])
            if not candidates:
                raise RuntimeError(f"no_candidates: {result.get('promptFeedback', result)}")

            parts = (candidates[0].get("content") or {}).get("parts") or []
            raw_text = "".join(
                part.get("text", "")
                for part in parts
                if isinstance(part, dict) and isinstance(part.get("text"), str)
            ).strip()
            if not raw_text:
                raise RuntimeError(f"empty response: {str(result)[:500]}")
            return raw_text, None
        except urllib.error.HTTPError as error:
            try:
                detail = error.read().decode("utf-8")[:500]
            except Exception:
                detail = str(error)
            wrapped_error = RuntimeError(f"HTTP {error.code}: {detail}")
            if attempt < args.max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                return None, str(wrapped_error)
        except Exception as error:
            if attempt < args.max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                return None, str(error)

    return None, "max_retries_exceeded"


def extract_json_object(text):
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start >= 0 and end > start:
        return json.loads(cleaned[start : end + 1])
    raise ValueError("No JSON object found in model output.")


def validate_result(result):
    errors = []
    if not isinstance(result, dict):
        return ["result is not a JSON object"]

    if not isinstance(result.get("has_cultural_meaning"), bool):
        errors.append("has_cultural_meaning must be boolean")

    if result.get("meaning_type") not in MEANING_TYPES:
        errors.append(f"meaning_type must be one of {sorted(MEANING_TYPES)}")

    if result.get("interpretation_confidence") not in CONFIDENCE_LEVELS:
        errors.append(f"interpretation_confidence must be one of {sorted(CONFIDENCE_LEVELS)}")

    for field in ["cultural_meaning", "cross_cultural_note"]:
        if not isinstance(result.get(field), str) or not result[field].strip():
            errors.append(f"{field} must be a non-empty string")

    if not isinstance(result.get("key_associations"), list):
        errors.append("key_associations must be a list")
    elif any(not isinstance(item, str) for item in result["key_associations"]):
        errors.append("key_associations must contain only strings")

    if isinstance(result.get("cultural_meaning"), str):
        sentence_count = len(re.findall(r"[.!?]", result["cultural_meaning"]))
        if sentence_count > 2:
            errors.append("cultural_meaning should be 1-2 sentences")
        if re.search(r"[\u4e00-\u9fff]", result["cultural_meaning"]):
            errors.append("cultural_meaning should be in English, not Chinese")

    if isinstance(result.get("cross_cultural_note"), str):
        if re.search(r"[\u4e00-\u9fff]", result["cross_cultural_note"]):
            errors.append("cross_cultural_note should be in English, not Chinese")

    if isinstance(result.get("key_associations"), list):
        if any(re.search(r"[\u4e00-\u9fff]", item) for item in result["key_associations"] if isinstance(item, str)):
            errors.append("key_associations should be in English, not Chinese")

    return errors


def load_done_ids(output_jsonl, failed_jsonl, retry_failed):
    done = set()
    for path in [output_jsonl]:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as file:
            for line in file:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                record_id = row.get("record_id")
                if record_id:
                    done.add(record_id)

    if not retry_failed and failed_jsonl.exists():
        with failed_jsonl.open("r", encoding="utf-8") as file:
            for line in file:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                record_id = row.get("record_id")
                if record_id:
                    done.add(record_id)
    return done


def load_failed_ids(failed_jsonl):
    failed_ids = set()
    if not failed_jsonl.exists():
        return failed_ids

    with failed_jsonl.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            record_id = row.get("record_id")
            if record_id:
                failed_ids.add(record_id)
    return failed_ids


def generate_one(record, args):
    prompt = build_prompt(record)
    raw_text, error = call_gemini_api(args.api_key, args.model, prompt, args)
    if raw_text is None:
        return None, {
            "record_id": record["record_id"],
            "concept": record["concept"],
            "dimension": record["dimension"],
            "culture": record["culture"],
            "error": error,
        }

    try:
        result = extract_json_object(raw_text)
        validation_errors = validate_result(result)
        if validation_errors:
            raise ValueError("; ".join(validation_errors))
    except Exception as parse_error:
        return None, {
            "record_id": record["record_id"],
            "concept": record["concept"],
            "dimension": record["dimension"],
            "culture": record["culture"],
            "error": f"parse_or_validation_failed: {parse_error}",
            "raw_text": raw_text,
        }

    entry = record["target_entry"]
    output = {
        "record_id": record["record_id"],
        "concept": record["concept"],
        "dimension": record["dimension"],
        "culture": record["culture"],
        "culture_label": CULTURE_LABELS.get(record["culture"], record["culture"]),
        **result,
        "evidence": {
            "target_distribution": entry.get("distribution", {}),
            "culture_baseline": record["culture_baseline"],
            "support_annotations": entry.get("support_annotations", 0),
            "support_images": entry.get("support_images", 0),
            "other_culture_distributions": {
                culture: other_entry.get("distribution", {})
                for culture, other_entry in record["other_entries"].items()
            },
            "example_texts_sample": record["example_texts"],
            "example_image_ids_sample": record["example_image_ids"],
        },
        "model": args.model,
    }
    return output, None


def append_jsonl(path, row):
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    args = get_args()
    input_kb_path = Path(args.input_kb)
    baseline_kb_path = Path(args.baseline_kb)
    output_jsonl = Path(args.output_jsonl)
    failed_jsonl = Path(args.failed_jsonl)
    retry_source_failed_jsonl = Path(args.retry_source_failed_jsonl) if args.retry_source_failed_jsonl else failed_jsonl
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    failed_jsonl.parent.mkdir(parents=True, exist_ok=True)

    cultures_to_run = get_cultures_arg(args.culture)
    kb = load_kb(input_kb_path)
    baseline_kb = load_kb(baseline_kb_path)
    records = build_records(kb, baseline_kb, cultures_to_run, args.include_example_texts)

    retry_failed = args.retry_failed or args.retry_failed_only
    done_ids = load_done_ids(output_jsonl, failed_jsonl, retry_failed)
    if args.retry_failed_only:
        failed_ids = load_failed_ids(retry_source_failed_jsonl)
        pending = [
            record
            for record in records
            if record["record_id"] in failed_ids and record["record_id"] not in done_ids
        ]
    else:
        failed_ids = set()
        pending = [record for record in records if record["record_id"] not in done_ids]
    if args.limit > 0:
        pending = pending[: args.limit]

    print(f"Input KB: {input_kb_path}")
    print(f"Baseline KB: {baseline_kb_path}")
    print(f"Output JSONL: {output_jsonl}")
    print(f"Failed JSONL: {failed_jsonl}")
    if args.retry_failed_only:
        print(f"Retry source failed JSONL: {retry_source_failed_jsonl}")
    print(f"Model: {args.model}")
    print(f"Endpoint: {NEWAPI_BASE_URL}/v1beta/models/{args.model}:generateContent")
    print(f"Total candidate records: {len(records)}")
    print(f"Already done/skipped: {len(done_ids)}")
    if args.retry_failed_only:
        print(f"Failed record_ids found: {len(failed_ids)}")
    print(f"Pending this run: {len(pending)}")
    print(f"Cultures: {', '.join(sorted(cultures_to_run))}")
    print(f"Workers: {args.workers}")

    if args.dry_run:
        print("\nDry run: no API calls will be made.")
        if pending:
            print("\nPrompt preview for first pending record:")
            print("=" * 80)
            print(build_prompt(pending[0])[:4000])
            print("=" * 80)
        return

    if not args.api_key:
        raise ValueError("Missing API key. Set NEWAPI_API_KEY or pass --api_key.")

    if not pending:
        print("Nothing to do.")
        return

    success = 0
    failed = 0
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        future_to_record = {
            pool.submit(generate_one, record, args): record
            for record in pending
        }
        completed_iter = as_completed(future_to_record)
        if tqdm is not None:
            progress = tqdm(completed_iter, total=len(future_to_record), unit="entry", dynamic_ncols=True)
        else:
            progress = completed_iter
        for future in progress:
            record = future_to_record[future]
            try:
                output, error = future.result()
            except Exception as exc:
                output = None
                error = {
                    "record_id": record["record_id"],
                    "concept": record["concept"],
                    "dimension": record["dimension"],
                    "culture": record["culture"],
                    "error": f"unexpected_exception: {exc}",
                }

            if output is not None:
                append_jsonl(output_jsonl, output)
                success += 1
            else:
                append_jsonl(failed_jsonl, error)
                failed += 1

            if args.sleep_interval > 0:
                time.sleep(args.sleep_interval)
            if tqdm is not None:
                progress.set_postfix({"ok": success, "fail": failed})
            elif (success + failed) % 50 == 0:
                print(f"  processed={success + failed}, ok={success}, fail={failed}")

    print("\nDone.")
    print(f"  Success: {success}")
    print(f"  Failed: {failed}")
    print(f"  Output: {output_jsonl}")
    print(f"  Failed rows: {failed_jsonl}")


if __name__ == "__main__":
    main()
