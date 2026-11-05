import argparse
import gc
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

parser = argparse.ArgumentParser()
parser.add_argument("model_id", type=str, help="Hugging Face model ID or local path")
parser.add_argument(
    "paths",
    nargs="+",
    help=(
        "Adapter mode: <adapter_path> <input_json> <output_json>. "
        "Base-model mode: <input_json> <output_json> with --base-model."
    ),
)
parser.add_argument(
    "--base-model",
    action="store_true",
    help="run the base model directly without loading a LoRA adapter",
)
parser.add_argument(
    "--seed", type=int, default=42, help="global random seed for deterministic inference"
)
parser.add_argument("--max_new_tokens", type=int, default=1024, help="maximum generated tokens")
args = parser.parse_args()

model_id = args.model_id
if args.base_model:
    if len(args.paths) != 2:
        parser.error("--base-model expects: model_id input_json output_json")
    adapter_path = None
    test_json, json_path = args.paths
else:
    if len(args.paths) != 3:
        parser.error(
            "adapter evaluation expects: model_id adapter_path input_json output_json; "
            "use --base-model for base-model evaluation"
        )
    adapter_path, test_json, json_path = args.paths
seed = args.seed
max_new_tokens = args.max_new_tokens


def set_global_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.allow_tf32 = False
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = False


set_global_seed(seed)
print(f"[InferenceConfig] seed={seed} max_new_tokens={max_new_tokens}")


def clear_memory():
    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        except Exception:
            pass
    print(f"GPU allocated memory: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    print(f"GPU reserved  memory: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")


def validate_image_paths(data: dict, json_path: Path) -> None:
    missing = []

    for sample_id, sample in data.items():
        if not isinstance(sample, dict):
            continue

        image_path = sample.get("image")
        if not image_path:
            missing.append(f"{sample_id}: missing image field")
            continue

        if not Path(str(image_path)).expanduser().exists():
            missing.append(f"{sample_id}: {image_path}")

    if missing:
        preview = "\n".join(missing[:5])
        raise FileNotFoundError(
            f"Could not find {len(missing)} image file(s) referenced by {json_path}. "
            f"Paths are checked exactly as written in the JSON.\n{preview}"
        )


def generate_text_from_sample(model, processor, sample, max_new_tokens=1024, device="cuda"):
    text_input = processor.apply_chat_template(
        sample["messages"][:2], tokenize=False, add_generation_prompt=True
    )

    image_inputs, _ = process_vision_info(sample["messages"])

    model_inputs = processor(
        text=[text_input],
        images=image_inputs,
        return_tensors="pt",
    ).to(device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        num_beams=1,
    )

    trimmed_generated_ids = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    output_text = processor.batch_decode(
        trimmed_generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    return output_text[0]


system_message = "You are a Vision-Language Model specialized in univariate time-series anomaly detection. Given a chart image and a prompt, identify whether an anomaly exists, indicate where it occurs, and briefly explain why it is anomalous. Keep your response clear, structured, and focused on visual evidence in the chart."


def format_data(sample):
    has_resp = "resp" in sample and sample["resp"] not in [None, ""]
    assistant_content = [{"type": "text", "text": sample["resp"] if has_resp else ""}]

    return {
        "images": [sample["image"]],
        "messages": [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_message}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": sample["image"]},
                    {"type": "text", "text": sample["query"]},
                ],
            },
            {
                "role": "assistant",
                "content": assistant_content,
            },
        ],
    }


test_json_path = Path(test_json)

with open(test_json_path, "r", encoding="utf-8") as f:
    test_data = json.load(f)

validate_image_paths(test_data, test_json_path)

sorted_test_items = sorted(test_data.items(), key=lambda x: x[0])
test_dataset = [(sample_id, format_data(sample)) for sample_id, sample in sorted_test_items]
print(f"[InferenceConfig] Loaded {len(test_dataset)} test samples from {test_json_path}")


clear_memory()


model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_id, device_map="auto", torch_dtype=torch.bfloat16
)
processor = AutoProcessor.from_pretrained(model_id)

if adapter_path is None:
    print("[InferenceConfig] Running base model without adapter")
else:
    model.load_adapter(adapter_path)
    print(f"[InferenceConfig] Loaded adapter from {adapter_path}")
Path(json_path).parent.mkdir(parents=True, exist_ok=True)

results = []


model.eval()
with torch.inference_mode():
    for i, (sample_id, sample) in enumerate(test_dataset):
        start_time = time.perf_counter()
        try:
            text_output = generate_text_from_sample(
                model,
                processor,
                sample,
                max_new_tokens=max_new_tokens,
            )
        except Exception as e:
            text_output = f"[ERROR] {e}"
        runtime = time.perf_counter() - start_time
        results.append(
            {
                "id": sample_id,
                "row_index": i,
                "image": sample["images"][0],
                "query": sample["messages"][1]["content"][1]["text"],
                "ground_truth": sample["messages"][2]["content"][0]["text"],
                "model_output": text_output,
                "runtime": runtime,
            }
        )

        if i % 10 == 0:
            print(f"Processed {i}/{len(test_dataset)} samples...")

            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            torch.cuda.empty_cache()
            time.sleep(1)


with open(json_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"Saved all outputs to {json_path}")
