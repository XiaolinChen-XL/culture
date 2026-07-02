import argparse
import csv
import importlib.util
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path


os.environ.setdefault("CUDA_VISIBLE_DEVICES", "4")
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable


SCRIPT_DIR = Path(__file__).resolve().parent
GEMINI_RAG_SCRIPT = SCRIPT_DIR / "gemini35_rag_emotion_eval-v2.py"
DEFAULT_MODEL_PATH = "/home/xiaolin/LLM/InternVL3_5-14B"
DEFAULT_TEST_CSV = "/home/xiaolin/dataset/ArtECulture/arteculture_6792_downsampled_with_country_nonblank_region_split_v2_test.csv"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "outputs_internvl35_14_concepts"
DEFAULT_IMAGE_BASE = "/home/xiaolin/dataset/ArtECulture/Image/"
OLD_IMAGE_PREFIX = "/home/xiaolin/dataset/ArtECulture/Image/"
OUTPUT_PREFIX = "internvl35_14"
CONCEPT_CACHE_NAME = f"{OUTPUT_PREFIX}_multilingual_image_concepts.jsonl"
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
    parser = argparse.ArgumentParser(description="Extract multilingual visual concepts with InternVL3.5-14B.")
    parser.add_argument("--test_csv", default=str(DEFAULT_TEST_CSV))
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--concept_cache_jsonl", default=None, help="Output/cache JSONL for InternVL concepts.")
    parser.add_argument("--image_base", default=str(DEFAULT_IMAGE_BASE))
    parser.add_argument("--old_image_prefix", default=OLD_IMAGE_PREFIX)
    parser.add_argument("--model_path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--max_samples", type=int, default=None, help="Limit number of test images for smoke tests.")
    parser.add_argument("--max_new_tokens", type=int, default=1024) # 2048
    parser.add_argument("--max_image_tiles", type=int, default=6) # 12
    parser.add_argument("--num_workers", type=int, default=0, help="CPU workers for image loading/preprocessing. 0 keeps the original inline path.")
    parser.add_argument("--retry_failed", action="store_true")
    parser.add_argument("--dry_run", action="store_true", help="Build inputs and write the concept prompt without loading InternVL.")
    return parser.parse_args()


def build_transform(input_size):
    import torchvision.transforms as transforms
    from torchvision.transforms.functional import InterpolationMode

    imagenet_mean = (0.485, 0.456, 0.406)
    imagenet_std = (0.229, 0.224, 0.225)
    return transforms.Compose(
        [
            transforms.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            transforms.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
        ]
    )


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height
    target_ratios = set(
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if min_num <= i * j <= max_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
    target_aspect_ratio = find_closest_aspect_ratio(aspect_ratio, target_ratios, orig_width, orig_height, image_size)
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        processed_images.append(resized_img.crop(box))
    if use_thumbnail and len(processed_images) != 1:
        processed_images.append(image.resize((image_size, image_size)))
    return processed_images


def load_image(image_file, torch_module, input_size=448, max_num=12):
    from PIL import Image

    image = Image.open(image_file).convert("RGB")
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(img) for img in images]
    return torch_module.stack(pixel_values)


def load_internvl_model(model_path):
    import torch
    from transformers import AutoModel, AutoTokenizer

    print("Loading InternVL3.5-14B model...")
    model = AutoModel.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        use_flash_attn=True,
        trust_remote_code=True,
    ).eval().cuda()
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        use_fast=False,
    )
    generation_config = {
        "max_new_tokens": None,
        "do_sample": False,
    }
    print(f"InternVL model loaded: {model_path}")
    return model, tokenizer, generation_config, torch


def call_internvl_model(model, tokenizer, generation_config, torch_module, image_path, prompt, max_new_tokens, max_image_tiles, pixel_values=None):
    try:
        question = prompt if prompt.lstrip().startswith("<image>") else "<image>\n" + prompt
        if pixel_values is None:
            pixel_values = load_image(image_path, torch_module, max_num=max_image_tiles)
        pixel_values = pixel_values.to(torch_module.bfloat16).cuda()
        local_generation_config = dict(generation_config)
        local_generation_config["max_new_tokens"] = max_new_tokens
        with torch_module.no_grad():
            response = model.chat(
                tokenizer,
                pixel_values,
                question,
                local_generation_config,
            )
        output_text = response.strip() if isinstance(response, str) else str(response).strip()
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


def has_any_concepts(obj):
    for culture in rag.CULTURES:
        culture_obj = obj.get(culture) or {}
        if culture_obj.get("caption"):
            return True
        for category in CONCEPT_CATEGORIES:
            if culture_obj.get(category):
                return True
    return False


def parse_partial_concept_bundle(text):
    cleaned = (text or "").strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`").strip()
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].strip()
    result = {}
    for index, culture in enumerate(rag.CULTURES):
        marker = f'"{culture}"'
        start = cleaned.find(marker)
        if start == -1:
            result[culture] = {}
            continue
        following_starts = [
            cleaned.find(f'"{next_culture}"', start + len(marker))
            for next_culture in rag.CULTURES[index + 1 :]
        ]
        following_starts = [pos for pos in following_starts if pos != -1]
        end = min(following_starts) if following_starts else len(cleaned)
        block = cleaned[start:end]
        culture_obj = {}
        caption_match = re.search(r'"caption"\s*:\s*"([^"]*)"', block, flags=re.DOTALL)
        if caption_match:
            culture_obj["caption"] = caption_match.group(1).strip()
        for category in CONCEPT_CATEGORIES:
            array_match = re.search(rf'"{re.escape(category)}"\s*:\s*\[(.*?)\]', block, flags=re.DOTALL)
            values = re.findall(r'"([^"]+)"', array_match.group(1)) if array_match else []
            culture_obj[category] = [value.strip() for value in values if value.strip()]
        result[culture] = culture_obj
    return result


def extract_concepts_with_internvl(model, tokenizer, generation_config, torch_module, image_file, image_path, max_new_tokens, max_image_tiles, pixel_values=None):
    raw_text, error = call_internvl_model(
        model,
        tokenizer,
        generation_config,
        torch_module,
        image_path,
        CONCEPT_EXTRACTION_PROMPT,
        max_new_tokens=max_new_tokens,
        max_image_tiles=max_image_tiles,
        pixel_values=pixel_values,
    )
    if raw_text is None:
        return None, f"concept_extract_failed: {error}"
    try:
        obj = parse_json_object(raw_text)
        return normalize_concept_bundle(obj, image_file, image_path, raw_text), None
    except Exception as parse_error:
        partial_obj = parse_partial_concept_bundle(raw_text)
        if has_any_concepts(partial_obj):
            record = normalize_concept_bundle(partial_obj, image_file, image_path, raw_text)
            record["parse_warning"] = f"partial_concept_json_recovered: {parse_error}"
            return record, None
        return None, f"concept_parse_failed: {parse_error}; raw={raw_text[:500]}"


def load_concept_cache(path, retry_failed=False):
    by_basename = {}
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return by_basename
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            obj = json.loads(line)
            if retry_failed and obj.get("status") != "OK":
                continue
            basename = rag.image_basename(obj.get("image_file") or obj.get("resolved_image_path"))
            if basename:
                by_basename[basename] = obj
    return by_basename


def append_jsonl(path, record):
    with Path(path).open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")


def count_concepts(concept_record):
    if not isinstance(concept_record, dict):
        return 0
    return sum(len(concept_record.get(category) or []) for category in CONCEPT_CATEGORIES)


def make_csv_row(row, basename, image_path, record):
    concepts = record.get("concepts") if isinstance(record, dict) else {}
    return {
        "image_file": row.get("image_file", ""),
        "image_basename": basename,
        "resolved_image_path": image_path or "",
        "country": row.get("country", ""),
        "region_group": row.get("region_group", ""),
        "status": record.get("status", "FAILED"),
        "error": record.get("error", ""),
        "parse_warning": record.get("parse_warning", ""),
        "english_concept_count": str(count_concepts((concepts or {}).get("english") or {})),
        "chinese_concept_count": str(count_concepts((concepts or {}).get("chinese") or {})),
        "arabic_concept_count": str(count_concepts((concepts or {}).get("arabic") or {})),
    }


def load_existing_csv_basenames(output_csv, retry_failed=False):
    existing = set()
    output_csv = Path(output_csv)
    if not output_csv.exists() or output_csv.stat().st_size == 0:
        return existing
    with output_csv.open("r", encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            if retry_failed and row.get("status") != "OK":
                continue
            basename = row.get("image_basename")
            if basename:
                existing.add(basename)
    return existing


class ConceptImageDataset:
    def __init__(self, rows, old_image_prefix, image_base, max_image_tiles):
        self.rows = rows
        self.old_image_prefix = old_image_prefix
        self.image_base = image_base
        self.max_image_tiles = max_image_tiles

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        import torch

        row = self.rows[index]
        image_path = rag.resolve_image_path(row.get("image_file"), self.old_image_prefix, self.image_base)
        item = {
            "row": row,
            "image_path": image_path or "",
            "pixel_values": None,
            "preprocess_error": "",
        }
        if not image_path:
            return item
        try:
            item["pixel_values"] = load_image(image_path, torch, max_num=self.max_image_tiles)
        except Exception as error:
            item["preprocess_error"] = str(error)
        return item


def iter_rows_for_inference(rows, args):
    if args.num_workers <= 0:
        for row in rows:
            image_path = rag.resolve_image_path(row.get("image_file"), args.old_image_prefix, args.image_base)
            yield {
                "row": row,
                "image_path": image_path or "",
                "pixel_values": None,
                "preprocess_error": "",
            }
        return

    from torch.utils.data import DataLoader

    dataset = ConceptImageDataset(rows, args.old_image_prefix, args.image_base, args.max_image_tiles)
    loader = DataLoader(
        dataset,
        batch_size=None,
        num_workers=args.num_workers,
        prefetch_factor=2,
        persistent_workers=args.num_workers > 0,
    )
    yield from loader


def main():
    args = get_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    concept_cache_jsonl = (
        Path(args.concept_cache_jsonl)
        if args.concept_cache_jsonl
        else output_dir / CONCEPT_CACHE_NAME
    )
    output_csv = output_dir / f"{OUTPUT_PREFIX}_concept_extraction_status.csv"
    summary_json = output_dir / f"{OUTPUT_PREFIX}_concept_extraction_summary.json"
    prompt_preview = output_dir / f"{OUTPUT_PREFIX}_concept_prompt_preview.txt"

    if args.dry_run:
        prompt_preview.write_text(CONCEPT_EXTRACTION_PROMPT, encoding="utf-8")
        print(f"Dry run concept prompt written: {prompt_preview}")
        print(CONCEPT_EXTRACTION_PROMPT[:4000])
        return

    test_rows = rag.load_test_rows(args.test_csv)
    if args.max_samples is not None:
        test_rows = test_rows[: args.max_samples]
    concepts_by_image = load_concept_cache(concept_cache_jsonl, retry_failed=args.retry_failed)
    existing_csv = load_existing_csv_basenames(output_csv, retry_failed=args.retry_failed)

    print(f"Input CSV: {args.test_csv}")
    print(f"Output dir: {output_dir}")
    print(f"Concept cache: {concept_cache_jsonl}")
    print(f"Samples: {len(test_rows)}")
    print(f"DataLoader workers: {args.num_workers}")

    model, tokenizer, generation_config, torch_module = load_internvl_model(args.model_path)

    header = [
        "image_file",
        "image_basename",
        "resolved_image_path",
        "country",
        "region_group",
        "status",
        "error",
        "parse_warning",
        "english_concept_count",
        "chinese_concept_count",
        "arabic_concept_count",
    ]
    file_exists = output_csv.exists() and output_csv.stat().st_size > 0
    status_counts = Counter()
    inference_rows = []
    for row in test_rows:
        basename = rag.image_basename(row.get("image_file"))
        if basename in existing_csv and basename in concepts_by_image:
            status_counts[concepts_by_image[basename].get("status", "OK")] += 1
        else:
            inference_rows.append(row)

    with output_csv.open("a", encoding="utf-8-sig", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=header)
        if not file_exists:
            writer.writeheader()

        pbar = tqdm(iter_rows_for_inference(inference_rows, args), total=len(inference_rows), desc=f"{OUTPUT_PREFIX} concept extraction", unit="image", dynamic_ncols=True)
        for item in pbar:
            row = item["row"]
            basename = rag.image_basename(row.get("image_file"))
            image_path = item["image_path"]
            if not image_path:
                record = {
                    "image_file": row.get("image_file"),
                    "resolved_image_path": "",
                    "raw_response": "",
                    "status": "FAILED",
                    "error": f"image_not_found: {row.get('image_file')}",
                    "concepts": {},
                }
            elif item.get("preprocess_error"):
                record = {
                    "image_file": row.get("image_file"),
                    "resolved_image_path": image_path,
                    "raw_response": "",
                    "status": "FAILED",
                    "error": f"image_preprocess_failed: {item.get('preprocess_error')}",
                    "concepts": {},
                }
            else:
                record, error = extract_concepts_with_internvl(
                    model,
                    tokenizer,
                    generation_config,
                    torch_module,
                    row.get("image_file"),
                    image_path,
                    max_new_tokens=args.max_new_tokens,
                    max_image_tiles=args.max_image_tiles,
                    pixel_values=item.get("pixel_values"),
                )
                if record is None:
                    record = {
                        "image_file": row.get("image_file"),
                        "resolved_image_path": image_path,
                        "raw_response": "",
                        "status": "FAILED",
                        "error": error or "",
                        "concepts": {},
                    }

            concepts_by_image[basename] = record
            append_jsonl(concept_cache_jsonl, record)
            writer.writerow(make_csv_row(row, basename, image_path, record))
            csv_file.flush()
            status_counts[record.get("status", "FAILED")] += 1
            pbar.set_postfix({"ok": status_counts.get("OK", 0), "fail": status_counts.get("FAILED", 0), "cache": len(concepts_by_image)})

    summary = {
        "model": args.model_path,
        "test_csv": str(args.test_csv),
        "concept_cache_jsonl": str(concept_cache_jsonl),
        "output_csv": str(output_csv),
        "max_samples": args.max_samples,
        "num_workers": args.num_workers,
        "max_new_tokens": args.max_new_tokens,
        "max_image_tiles": args.max_image_tiles,
        "status_counts": dict(status_counts),
        "cached_records": len(concepts_by_image),
    }
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote concept cache: {concept_cache_jsonl}")
    print(f"Wrote status CSV: {output_csv}")
    print(f"Wrote summary: {summary_json}")


if __name__ == "__main__":
    main()
