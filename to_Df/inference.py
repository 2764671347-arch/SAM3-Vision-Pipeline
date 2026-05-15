#!/usr/bin/env python3
"""
SAM3 推理脚本
基于 test_task.json 的文本提示进行 SAM3 推理，输出比赛提交格式 predictions.json。

使用方式：
1. 本文件顶部的 DEFAULT_* 常量不可修改
2. 运行 `python inference.py`
"""

import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image


from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

try:
    from pycocotools import mask as maskUtils
except ImportError:
    print("警告: pycocotools 未安装，将使用备用 RLE 编码方法")
    maskUtils = None


DEFAULT_TASKS = "/raytron/test/test_tasks.json"
DEFAULT_IMAGE_ROOT = "/raytron/test/"
DEFAULT_OUTPUT_PATH = "/raytron/test/predictions.json"
DEFAULT_CHECKPOINT_PATH = "/raytron/code/model/sam3.pt"
DEFAULT_CONF_THRESHOLD = 0.01


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_model(checkpoint_path: Optional[str]):
    if checkpoint_path and Path(checkpoint_path).exists():
        model = build_sam3_image_model(checkpoint_path=checkpoint_path, device=DEVICE)
    else:
        model = build_sam3_image_model(device=DEVICE)
    processor = Sam3Processor(model=model, device=DEVICE)
    return model, processor


def count_model_params(model) -> Dict[str, Any]:
    total_params = sum(param.numel() for param in model.parameters())
    trainable_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
    return {
        "device": DEVICE,
        "total_params": int(total_params),
        "trainable_params": int(trainable_params),
    }


def simple_rle_encode(binary_mask: np.ndarray) -> Dict[str, Any]:
    flat = binary_mask.astype(np.uint8).flatten(order="F")
    runs: List[int] = []
    prev = 0
    count = 0

    for value in flat:
        value = int(value)
        if value == prev:
            count += 1
        else:
            runs.append(count)
            count = 1
            prev = value
    runs.append(count)

    return {
        "size": [int(binary_mask.shape[0]), int(binary_mask.shape[1])],
        "counts": ",".join(map(str, runs)),
    }


def mask_to_rle(binary_mask: np.ndarray) -> Dict[str, Any]:
    binary_mask = binary_mask.astype(np.uint8)
    if maskUtils is None:
        return simple_rle_encode(binary_mask)

    rle = maskUtils.encode(np.asfortranarray(binary_mask))
    if isinstance(rle, list):
        rle = rle[0]

    return {
        "size": [int(rle["size"][0]), int(rle["size"][1])],
        "counts": rle["counts"].decode("utf-8")
        if isinstance(rle["counts"], bytes)
        else rle["counts"],
    }


def build_empty_rle(height: int, width: int) -> Dict[str, Any]:
    return mask_to_rle(np.zeros((height, width), dtype=np.uint8))


def normalize_mask(mask_tensor: torch.Tensor) -> np.ndarray:
    mask = mask_tensor.detach().cpu().numpy()
    if mask.ndim == 3:
        mask = mask[0]
    return (mask > 0).astype(np.uint8)


def aggregate_prompt_prediction(
    state: Dict[str, Any], prompt: str, conf_threshold: float
) -> Optional[Dict[str, Any]]:
    masks = state.get("masks")
    scores = state.get("scores")

    if masks is None or len(masks) == 0:
        return None

    if scores is None or len(scores) == 0:
        scores_np = np.zeros(len(masks), dtype=np.float32)
    else:
        scores_np = scores.detach().cpu().numpy().reshape(-1)

    selected_indices = [
        index for index, score in enumerate(scores_np) if float(score) >= conf_threshold
    ]
    if not selected_indices:
        return None

    merged_mask: Optional[np.ndarray] = None
    for index in selected_indices:
        mask_np = normalize_mask(masks[index])
        if merged_mask is None:
            merged_mask = mask_np
        else:
            merged_mask = np.logical_or(merged_mask, mask_np).astype(np.uint8)

    if merged_mask is None:
        return None

    return {
        "prompt": prompt,
        "score": float(np.max(scores_np[selected_indices])) if len(scores_np) > 0 else 0.0,
        "instance_count": len(selected_indices),
        "rle": mask_to_rle(merged_mask),
    }


@torch.inference_mode()
def do_inference(
    image_path: str,
    text_prompts: List[str],
    processor,
    conf_threshold: float,
) -> Tuple[Dict[str, Dict[str, Any]], int, int]:
    image = Image.open(image_path).convert("RGB")
    width, height = image.size

    state = processor.set_image(image)
    results_by_prompt: Dict[str, Dict[str, Any]] = {}

    for prompt in text_prompts:
        processor.reset_all_prompts(state)
        state = processor.set_text_prompt(prompt=prompt, state=state)
        prompt_prediction = aggregate_prompt_prediction(state, prompt, conf_threshold)
        if prompt_prediction is not None:
            results_by_prompt[prompt] = prompt_prediction

    return results_by_prompt, width, height


def load_tasks(tasks_path: str) -> List[Dict[str, Any]]:
    with open(tasks_path, "r", encoding="utf-8") as file_obj:
        tasks = json.load(file_obj)

    if not isinstance(tasks, list):
        raise ValueError("任务文件必须是 JSON 数组")

    for index, task in enumerate(tasks):
        if not isinstance(task, dict):
            raise ValueError(f"第 {index} 条任务不是 JSON 对象")
        if "ann_id" not in task or "image_path" not in task or "text_prompt" not in task:
            raise ValueError(f"第 {index} 条任务缺少 ann_id/image_path/text_prompt 字段")

    return tasks


def process_tasks(
    tasks: List[Dict[str, Any]],
    image_root: str,
    checkpoint_path: Optional[str],
    output_path: str,
    conf_threshold: float,
) -> None:
    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)

    model, processor = load_model(checkpoint_path)
    model_info = count_model_params(model)
    if checkpoint_path:
        model_info["checkpoint_path"] = checkpoint_path

    tasks_by_image: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for task in tasks:
        tasks_by_image[task["image_path"]].append(task)

    print(f"任务总数: {len(tasks)}")
    print(f"图片总数: {len(tasks_by_image)}")

    processed_images = 0
    inference_total_time = 0.0
    task_to_rle: Dict[Any, Dict[str, Any]] = {}

    for image_rel_path, image_tasks in tasks_by_image.items():
        image_abs_path = os.path.join(image_root, image_rel_path)
        print(f"\n处理图片: {image_rel_path}")
        print(f"  绝对路径: {image_abs_path}")

        if not os.path.exists(image_abs_path):
            raise FileNotFoundError(f"图片不存在: {image_abs_path}")

        unique_prompts = list(
            dict.fromkeys(
                task.get("text_prompt", "").strip()
                for task in image_tasks
                if task.get("text_prompt", "").strip()
            )
        )
        print(f"  文本提示: {unique_prompts}")

        image_start = time.time()
        results_by_prompt, width, height = do_inference(
            image_path=image_abs_path,
            text_prompts=unique_prompts,
            processor=processor,
            conf_threshold=conf_threshold,
        )
        empty_rle = build_empty_rle(height, width)

        for task in image_tasks:
            ann_id = task["ann_id"]
            prompt = task.get("text_prompt", "").strip()
            prediction = results_by_prompt.get(prompt)
            task_to_rle[ann_id] = prediction["rle"] if prediction is not None else empty_rle

        elapsed = time.time() - image_start
        inference_total_time += elapsed
        processed_images += 1
        hit_instances = sum(
            result.get("instance_count", 0) for result in results_by_prompt.values()
        )
        print(
            f"  完成: {len(results_by_prompt)}/{len(unique_prompts)} 个 prompt 命中, 合并 {hit_instances} 个实例, 耗时 {elapsed:.2f}s"
        )

    predictions_output = []
    for task in tasks:
        ann_id = task["ann_id"]
        if ann_id not in task_to_rle:
            raise RuntimeError(f"ann_id {ann_id} 未生成结果")
        predictions_output.append({"ann_id": ann_id, "rle": task_to_rle[ann_id]})

    avg_inference_time = (
        inference_total_time / processed_images if processed_images > 0 else 0.0
    )

    output_data = {
        "model_info": model_info,
        "timing": {
            "inference_seconds": float(inference_total_time),
            "avg_inference_seconds_per_image": float(avg_inference_time),
            "processed_images": processed_images,
            "total_tasks": len(tasks),
        },
        "predictions": predictions_output,
    }

    with open(output_path_obj, "w", encoding="utf-8") as file_obj:
        json.dump(output_data, file_obj, ensure_ascii=False, indent=2)

    print("\n推理完成!")
    print(f"输出文件: {output_path_obj}")
    print(f"纯推理总耗时: {inference_total_time:.2f}s")
    if processed_images > 0:
        print(f"平均每张图推理耗时: {avg_inference_time:.2f}s")


def main() -> None:
    print("=" * 60)
    print("SAM3 推理配置")
    print("=" * 60)
    print(f"任务文件: {DEFAULT_TASKS}")
    print(f"图片根目录: {DEFAULT_IMAGE_ROOT}")
    print(f"输出路径: {DEFAULT_OUTPUT_PATH}")
    print(f"模型检查点: {DEFAULT_CHECKPOINT_PATH}")
    print(f"置信度阈值: {DEFAULT_CONF_THRESHOLD}")
    print(f"设备: {DEVICE}")
    print("=" * 60)

    tasks = load_tasks(DEFAULT_TASKS)
    process_tasks(
        tasks=tasks,
        image_root=DEFAULT_IMAGE_ROOT,
        checkpoint_path=DEFAULT_CHECKPOINT_PATH,
        output_path=DEFAULT_OUTPUT_PATH,
        conf_threshold=DEFAULT_CONF_THRESHOLD,
    )


if __name__ == "__main__":
    main()
