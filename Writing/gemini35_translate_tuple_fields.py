import argparse
import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path


NEWAPI_BASE_URL = "https://www.dmxapi.cn"
DEFAULT_MODEL = "gemini-3.5-flash"

DEFAULT_AR_INPUT =  "/home/xiaolin/Model_35_ArtElingo/concept_culture_association_emotion_distribution_tuples_gemini35_english_ar.json"
DEFAULT_ZH_INPUT = "/home/xiaolin/Model_35_ArtElingo/concept_culture_association_emotion_distribution_tuples_gemini35_english_zh.json"

TARGETS = {
    "ar": {
        "language": "Modern Standard Arabic",
        "concept_field": "concept_ar",
        "associations_field": "key_associations_ar",
        "input": DEFAULT_AR_INPUT,
    },
    "zh": {
        "language": "Simplified Chinese",
        "concept_field": "concept_zh",
        "associations_field": "key_associations_zh",
        "input": DEFAULT_ZH_INPUT,
    },
}


def get_args():
    parser = argparse.ArgumentParser(
        description="Translate concept and key_associations fields in tuple JSON files using Gemini 3.5."
    )
    parser.add_argument("--culture", default="all", help="all, ar, zh, or comma-separated list like ar,zh")
    parser.add_argument("--ar_input", default=str(DEFAULT_AR_INPUT))
    parser.add_argument("--zh_input", default=str(DEFAULT_ZH_INPUT))
    parser.add_argument("--api_key", default=os.environ.get("NEWAPI_API_KEY", "sk-wGOeW1GNPpb3nWMP2yoXbz4wktH0DM3cVwr85Ab5vyl4TKxj"))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--batch_size", type=int, default=80)
    parser.add_argument("--max_retries", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--sleep_interval", type=float, default=0.05)
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def parse_cultures(raw):
    if raw.strip().lower() == "all":
        return ["ar", "zh"]
    cultures = [item.strip().lower() for item in raw.split(",") if item.strip()]
    invalid = [culture for culture in cultures if culture not in TARGETS]
    if invalid:
        raise ValueError(f"Invalid culture(s): {invalid}. Use ar, zh, or all.")
    return cultures


def load_tuple_file(path):
    with Path(path).open("r", encoding="utf-8") as file:
        payload = json.load(file)
    tuples = payload.get("tuples")
    if not isinstance(tuples, list):
        raise ValueError(f"Missing list field 'tuples' in {path}")
    return payload, tuples


def collect_unique_strings(tuples):
    values = set()
    for row in tuples:
        concept = row.get("concept")
        if isinstance(concept, str) and concept.strip():
            values.add(concept.strip())
        for item in row.get("key_associations") or []:
            if isinstance(item, str) and item.strip():
                values.add(item.strip())
    return sorted(values)


def chunked(items, size):
    for index in range(0, len(items), size):
        yield items[index : index + size]


def build_prompt(strings, target_language):
    numbered = "\n".join(f"{idx + 1}. {value}" for idx, value in enumerate(strings))
    return f"""Translate the following visual-art concept labels and short association phrases into {target_language}.

Rules:
- Preserve the meaning in a concise natural phrase.
- Keep proper nouns when translation would be awkward, but translate common descriptive words.
- Do not explain.
- Return valid JSON only.
- The JSON must map each exact English source string to its translation.

English source strings:
{numbered}

Required JSON format:
{{
  "source string 1": "translation 1",
  "source string 2": "translation 2"
}}
"""


def call_gemini(api_key, model, prompt, args):
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.0,
            "maxOutputTokens": 8192,
            "thinkingConfig": {"thinkingBudget": 0},
        },
    }
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    url = f"{NEWAPI_BASE_URL}/v1beta/models/{model}:generateContent?key={api_key}"
    headers = {"Content-Type": "application/json"}

    for attempt in range(args.max_retries):
        try:
            request = urllib.request.Request(url, data=body, headers=headers, method="POST")
            with urllib.request.urlopen(request, timeout=args.timeout) as response:
                response_body = response.read().decode("utf-8")
            result = json.loads(response_body)
            candidates = result.get("candidates") or []
            if not candidates:
                raise RuntimeError(f"no_candidates: {result.get('promptFeedback', result)}")
            parts = (candidates[0].get("content") or {}).get("parts") or []
            text = "".join(part.get("text", "") for part in parts if isinstance(part, dict)).strip()
            if not text:
                raise RuntimeError(f"empty response: {str(result)[:300]}")
            return text, None
        except urllib.error.HTTPError as error:
            try:
                detail = error.read().decode("utf-8")[:500]
            except Exception:
                detail = str(error)
            last_error = f"HTTP {error.code}: {detail}"
        except Exception as error:
            last_error = str(error)

        if attempt < args.max_retries - 1:
            time.sleep(2 ** attempt)

    return None, last_error


def extract_json(text):
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start >= 0 and end > start:
            return json.loads(cleaned[start : end + 1])
    raise ValueError("No JSON object found in response")


def load_cache(path):
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise ValueError(f"Translation cache must be a JSON object: {path}")
    return data


def save_cache(path, cache):
    with path.open("w", encoding="utf-8") as file:
        json.dump(cache, file, ensure_ascii=False, indent=2)


def translate_strings(strings, target_language, cache_path, args):
    cache = load_cache(cache_path)
    pending = [value for value in strings if value not in cache]
    print(f"  cache: {len(cache)} translated, pending: {len(pending)}")

    if args.dry_run:
        print("  dry_run: no API calls")
        return cache
    if not args.api_key:
        raise ValueError("Missing API key. Set NEWAPI_API_KEY or pass --api_key.")

    for batch_idx, batch in enumerate(chunked(pending, args.batch_size), 1):
        prompt = build_prompt(batch, target_language)
        raw, error = call_gemini(args.api_key, args.model, prompt, args)
        if raw is None:
            print(f"  batch {batch_idx} failed: {error}")
            continue
        try:
            mapping = extract_json(raw)
        except Exception as parse_error:
            print(f"  batch {batch_idx} parse failed: {parse_error}")
            continue
        added = 0
        for source in batch:
            translation = mapping.get(source)
            if isinstance(translation, str) and translation.strip():
                cache[source] = translation.strip()
                added += 1
        save_cache(cache_path, cache)
        print(f"  batch {batch_idx}: added {added}/{len(batch)}, cache={len(cache)}")
        if args.sleep_interval > 0:
            time.sleep(args.sleep_interval)
    return cache


def enrich_payload(payload, tuples, culture, cache, output_path):
    concept_field = TARGETS[culture]["concept_field"]
    associations_field = TARGETS[culture]["associations_field"]
    enriched = []
    missing = 0
    for row in tuples:
        new_row = dict(row)
        concept = row.get("concept")
        concept_translation = cache.get(concept) if isinstance(concept, str) else None
        if not concept_translation:
            missing += 1
        new_row[concept_field] = concept_translation
        translated_associations = []
        for item in row.get("key_associations") or []:
            translated = cache.get(item)
            if not translated:
                missing += 1
            translated_associations.append(translated)
        new_row[associations_field] = translated_associations
        enriched.append(new_row)

    output_payload = dict(payload)
    output_payload["translation_language"] = TARGETS[culture]["language"]
    output_payload["translation_fields"] = {
        "concept": concept_field,
        "key_associations": associations_field,
    }
    output_payload["translation_missing_value_count"] = missing
    output_payload["tuples"] = enriched
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(output_payload, file, ensure_ascii=False, indent=2)
    return missing


def main():
    args = get_args()
    input_paths = {
        "ar": Path(args.ar_input),
        "zh": Path(args.zh_input),
    }

    for culture in parse_cultures(args.culture):
        input_path = input_paths[culture]
        target_language = TARGETS[culture]["language"]
        output_path = input_path.with_name(input_path.stem + f"_with_{culture}_translations.json")
        cache_path = input_path.with_name(input_path.stem + f"_{culture}_translation_cache.json")

        print(f"\nCulture: {culture} -> {target_language}")
        print(f"  input: {input_path}")
        print(f"  output: {output_path}")
        print(f"  cache: {cache_path}")
        payload, tuples = load_tuple_file(input_path)
        strings = collect_unique_strings(tuples)
        print(f"  rows: {len(tuples)}, unique strings: {len(strings)}")
        cache = translate_strings(strings, target_language, cache_path, args)
        if args.dry_run:
            continue
        missing = enrich_payload(payload, tuples, culture, cache, output_path)
        print(f"  saved: {output_path}")
        print(f"  missing translated values: {missing}")


if __name__ == "__main__":
    main()
