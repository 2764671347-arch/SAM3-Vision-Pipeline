#!/usr/bin/env python3

import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Clean COCO instance-segmentation data for supervised SAM3 training."
    )
    parser.add_argument("--input-json", required=True, help="Input COCO annotation JSON")
    parser.add_argument(
        "--output-json", required=True, help="Path for the cleaned COCO annotation JSON"
    )
    parser.add_argument(
        "--prompts-output",
        default=None,
        help="Optional JSON path for category prompt variants",
    )
    parser.add_argument(
        "--aliases-json",
        default=None,
        help="Optional JSON mapping category id/name to alias strings or string lists",
    )
    parser.add_argument(
        "--category-whitelist",
        default=None,
        help="Comma-separated category ids or names to keep",
    )
    parser.add_argument(
        "--min-area",
        type=float,
        default=0.0,
        help="Drop annotations whose area is smaller than this value",
    )
    parser.add_argument(
        "--drop-crowd",
        action="store_true",
        help="Drop annotations with iscrowd=1",
    )
    parser.add_argument(
        "--remap-category-ids",
        action="store_true",
        help="Remap kept category ids to a contiguous 1..N range",
    )
    parser.add_argument(
        "--drop-images-without-annotations",
        action="store_true",
        help="Drop images that become annotation-free after filtering",
    )
    return parser.parse_args()


def load_json(path):
    with open(path, "r", encoding="utf-8") as fopen:
        return json.load(fopen)


def normalize_aliases(alias_data):
    if alias_data is None:
        return {}
    normalized = {}
    for key, value in alias_data.items():
        values = [value] if isinstance(value, str) else list(value)
        cleaned_values = []
        for item in values:
            if not isinstance(item, str):
                continue
            stripped = item.strip()
            if stripped:
                cleaned_values.append(stripped)
        values = cleaned_values
        if values:
            normalized[str(key)] = list(dict.fromkeys(values))
    return normalized


def normalize_whitelist(raw_value, categories):
    if raw_value is None:
        return None
    keep_tokens = {token.strip() for token in raw_value.split(",") if token.strip()}
    keep_ids = set()
    for category in categories:
        cat_id = str(category["id"])
        cat_name = str(category["name"])
        if cat_id in keep_tokens or cat_name in keep_tokens:
            keep_ids.add(category["id"])
    return keep_ids


def has_valid_segmentation(annotation):
    segmentation = annotation.get("segmentation")
    if segmentation is None:
        return False
    if isinstance(segmentation, list):
        return any(segment for segment in segmentation)
    if isinstance(segmentation, dict):
        counts = segmentation.get("counts")
        return counts not in (None, [], "")
    return False


def build_prompt_metadata(categories, aliases):
    prompts = {}
    for category in categories:
        cat_id = category["id"]
        original_id = category.get("_original_id", cat_id)
        names = [category["name"]]
        names.extend(aliases.get(str(cat_id), []))
        names.extend(aliases.get(str(original_id), []))
        names.extend(aliases.get(str(category["name"]), []))
        prompts[str(cat_id)] = list(dict.fromkeys(name for name in names if name))
    return prompts


def main():
    args = parse_args()
    coco = load_json(args.input_json)
    aliases = normalize_aliases(load_json(args.aliases_json) if args.aliases_json else None)

    images = coco.get("images", [])
    categories = coco.get("categories", [])
    annotations = coco.get("annotations", [])

    keep_category_ids = normalize_whitelist(args.category_whitelist, categories)
    categories = [
        category
        for category in categories
        if keep_category_ids is None or category["id"] in keep_category_ids
    ]
    category_ids = {category["id"] for category in categories}
    image_ids = {image["id"] for image in images}

    filtered_annotations = []
    kept_image_ids = set()
    for annotation in annotations:
        if annotation.get("image_id") not in image_ids:
            continue
        if annotation.get("category_id") not in category_ids:
            continue
        if args.drop_crowd and int(annotation.get("iscrowd", 0)) == 1:
            continue
        if not has_valid_segmentation(annotation):
            continue
        area = float(annotation.get("area", 0.0) or 0.0)
        if area <= 0.0:
            bbox = annotation.get("bbox") or [0, 0, 0, 0]
            if len(bbox) == 4:
                bbox_width = float(bbox[2])
                bbox_height = float(bbox[3])
                if bbox_width > 0.0 and bbox_height > 0.0:
                    area = float(bbox_width * bbox_height)
        if area < args.min_area:
            continue

        annotation = dict(annotation)
        annotation["area"] = area
        annotation["iscrowd"] = int(annotation.get("iscrowd", 0))
        filtered_annotations.append(annotation)
        kept_image_ids.add(annotation["image_id"])

    if args.drop_images_without_annotations:
        images = [image for image in images if image["id"] in kept_image_ids]

    if args.remap_category_ids:
        # COCO category ids are conventionally 1-indexed, so we keep that layout after filtering.
        remap = {category["id"]: index for index, category in enumerate(categories, start=1)}
        remapped_categories = []
        for category in categories:
            category = dict(category)
            category["_original_id"] = category["id"]
            category["id"] = remap[category["id"]]
            remapped_categories.append(category)
        categories = remapped_categories
        for annotation in filtered_annotations:
            annotation["category_id"] = remap[annotation["category_id"]]

    output_categories = [
        {key: value for key, value in category.items() if key != "_original_id"}
        for category in categories
    ]

    output = {
        "images": images,
        "annotations": filtered_annotations,
        "categories": output_categories,
    }
    if "info" in coco:
        output["info"] = coco["info"]
    if "licenses" in coco:
        output["licenses"] = coco["licenses"]

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fopen:
        json.dump(output, fopen, ensure_ascii=False)

    if args.prompts_output:
        prompts = build_prompt_metadata(categories, aliases)
        prompts_path = Path(args.prompts_output)
        prompts_path.parent.mkdir(parents=True, exist_ok=True)
        with open(prompts_path, "w", encoding="utf-8") as fopen:
            json.dump(prompts, fopen, ensure_ascii=False, indent=2)

    print(
        json.dumps(
            {
                "images": len(images),
                "annotations": len(filtered_annotations),
                "categories": len(categories),
                "output_json": str(output_path),
                "prompts_output": args.prompts_output,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
