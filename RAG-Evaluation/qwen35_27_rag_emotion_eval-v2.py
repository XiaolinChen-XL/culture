import argparse
import csv
import importlib.util
import json
import os
import time
from collections import Counter
from pathlib import Path


os.environ.setdefault("CUDA_VISIBLE_DEVICES", "5")

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable


SCRIPT_DIR = Path(__file__).resolve().parent
GEMINI_RAG_SCRIPT = SCRIPT_DIR / "gemini35_rag_emotion_eval-v2.py"
DEFAULT_MODEL_PATH = "/home/xiaolin/LLM/Qwen3.5-27B"
OUTPUT_PREFIX = "qwen35_27"
CONCEPT_CACHE_NAME = f"{OUTPUT_PREFIX}_image_concepts.jsonl"
CONCEPT_CATEGORIES = [
    "objects_subjects",
    "colors_color_schemes",
    "compositional_style",
    "scenes_settings",
    "symbolic_cultural_motifs",
]

CONCEPT_EXTRACTION_PROMPT = """
Analyze this painting and extract concise English visual concepts for later cultural-emotion retrieval.

Return ONLY one valid JSON object with exactly these keys:
{
  "caption": "one sentence visual caption",
  "objects_subjects": ["short noun phrase", "..."],
  "colors_color_schemes": ["short color or palette phrase", "..."],
  "compositional_style": ["short style/composition phrase", "..."],
  "scenes_settings": ["short scene or setting phrase", "..."],
  "symbolic_cultural_motifs": ["short symbolic or cultural motif phrase", "..."]
}

Rules:
- Use English only.
- Keep each concept short, usually 1-5 words.
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
    parser.add_argument("--test_csv", default=str(rag.DEFAULT_TEST_CSV))
    parser.add_argument("--concept_jsonl", default=None, help="Deprecated; Qwen concepts are extracted online and cached.")
    parser.add_argument("--concept_cache_jsonl", default=None, help="Cache for Qwen-extracted image concepts.")
    parser.add_argument("--kb_dir", default=str(rag.DEFAULT_KB_DIR))
    parser.add_argument("--output_dir", default=str(rag.DEFAULT_OUTPUT_DIR))
    parser.add_argument("--image_base", default=str(rag.DEFAULT_IMAGE_BASE))
    parser.add_argument("--old_image_prefix", default=rag.OLD_IMAGE_PREFIX)
    parser.add_argument("--model_path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--max_samples", type=int, default=None, help="Limit number of test images for smoke tests.")
    parser.add_argument("--top_k", type=int, default=6)
    parser.add_argument("--cultural_only", action="store_true", help="Retrieve only KB entries with has_cultural_meaning=true.")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--max_concept_tokens", type=int, default=512)
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


def normalize_concept_record(obj, image_file, image_path, raw_response):
    record = {
        "image_file": image_file,
        "resolved_image_path": image_path,
        "raw_response": raw_response,
        "caption": str(obj.get("caption") or "").strip(),
        "status": "OK",
        "error": "",
    }
    for category in CONCEPT_CATEGORIES:
        values = obj.get(category) or []
        if not isinstance(values, list):
            values = [values]
        record[category] = [str(value).strip() for value in values if str(value).strip()]
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
        return normalize_concept_record(obj, image_file, image_path, raw_text), None
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
        kb_records = rag.load_kb_records(args.kb_dir)
        kb_index = rag.build_kb_index(kb_records)

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
            concept_record = None
            concept_error = ""

            if args.mode in ("concept", "rag"):
                concept_record = concepts_by_image.get(basename)
                if concept_record is None and args.dry_run:
                    concept_prompt_preview.write_text(CONCEPT_EXTRACTION_PROMPT, encoding="utf-8")
                    print(f"Dry run concept prompt written: {concept_prompt_preview}")
                elif concept_record is None and not image_path:
                    concept_error = f"image_not_found_for_concept: {row.get('image_file')}"
                elif concept_record is None:
                    concept_record, concept_error = extract_concepts_with_qwen(
                        model,
                        processor,
                        device,
                        torch_module,
                        row.get("image_file"),
                        image_path,
                        max_new_tokens=args.max_concept_tokens,
                    )
                    if concept_record is not None:
                        concepts_by_image[basename] = concept_record
                        append_concept_cache(concept_cache_jsonl, concept_record)
                flat_concepts = rag.flatten_concepts(concept_record)
            else:
                flat_concepts = []

            for culture in cultures:
                key = (row.get("image_file"), culture, args.mode)
                if key in existing:
                    continue

                gt_emotion = row.get(f"{culture}_emotion", "")
                retrieved = []
                if args.mode == "rag":
                    retrieved = rag.retrieve_kb(
                        flat_concepts,
                        culture,
                        kb_index,
                        top_k=args.top_k,
                        include_visual_affective=not args.cultural_only,
                    )
                prompt = rag.build_prompt(args.mode, culture, concept_record, retrieved)

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
                    pred_emotion = rag.parse_emotion(raw_text)
                    reason = rag.parse_reason(raw_text)
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
                    "concept_status": "OK" if concept_record else ("FAILED" if concept_error else "SKIPPED"),
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
        "metrics": rag.compute_metrics(predictions),
        "status_counts": dict(Counter(r.get("status") for r in predictions)),
    }
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote predictions: {output_csv}")
    print(f"Wrote details: {output_jsonl}")
    print(f"Wrote summary: {summary_json}")


if __name__ == "__main__":
    main()
