import argparse
from datetime import datetime, timezone
import gc
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import platform
import random
import sys

DEFAULT_NUM_EPOCHS = 6
DEFAULT_LEARNING_RATE = 6.4e-4


parser = argparse.ArgumentParser()
parser.add_argument("model_id", type=str, help="Hugging Face model ID or local path")
parser.add_argument("output_dir", type=str, help="Output directory for saving model")
parser.add_argument("train_json", type=str, help="Input training JSON path")
parser.add_argument(
    "--num_epochs",
    "--num-epochs",
    dest="num_epochs",
    type=int,
    default=DEFAULT_NUM_EPOCHS,
    help=f"number of training epochs (default: {DEFAULT_NUM_EPOCHS})",
)
parser.add_argument(
    "--learn_rate",
    "--learn-rate",
    dest="learn_rate",
    type=float,
    default=DEFAULT_LEARNING_RATE,
    help=f"learning rate (default: {DEFAULT_LEARNING_RATE})",
)
parser.add_argument("--seed", type=int, default=42, help="global random seed")
parser.add_argument("--val_ratio", type=float, default=0.1, help="validation split ratio")
parser.add_argument(
    "--val_size",
    type=int,
    default=None,
    help="fixed validation size (overrides --val_ratio if provided)",
)
parser.add_argument(
    "--repro_mode",
    type=str,
    choices=["strict", "best_effort"],
    default="strict",
    help=(
        "Reproducibility profile. strict enforces deterministic kernels and "
        "disables known nondeterministic choices."
    ),
)
parser.add_argument(
    "--load_in_4bit",
    action="store_true",
    help=(
        "Enable 4-bit bitsandbytes quantization (lower VRAM), which is not "
        "compatible with strict determinism."
    ),
)
parser.add_argument(
    "--optim_name",
    type=str,
    choices=["adamw_torch", "adamw_torch_fused"],
    default=None,
    help="Optional optimizer override. If unset, selected from repro_mode.",
)
parser.add_argument(
    "--eval_strategy_mode",
    type=str,
    choices=["auto", "steps", "epoch"],
    default="auto",
    help=(
        "How to schedule evaluation/checkpointing. "
        "auto picks epoch for short runs and steps for longer runs."
    ),
)
parser.add_argument(
    "--evals_per_epoch",
    type=int,
    default=4,
    help="Target number of evaluations per epoch when step-based eval is used.",
)


args = parser.parse_args()

model_id = args.model_id
output_dir = args.output_dir
train_json = args.train_json
num_epoch = args.num_epochs
lr = args.learn_rate
seed = args.seed
val_ratio = args.val_ratio
val_size_arg = args.val_size
repro_mode = args.repro_mode
strict_determinism = repro_mode == "strict"
load_in_4bit = args.load_in_4bit
optim_name = args.optim_name or ("adamw_torch" if strict_determinism else "adamw_torch_fused")
eval_strategy_mode = args.eval_strategy_mode
evals_per_epoch = args.evals_per_epoch

if strict_determinism and load_in_4bit:
    raise ValueError(
        "Strict determinism is incompatible with --load_in_4bit because "
        "bitsandbytes 4-bit kernels are nondeterministic."
    )
if strict_determinism and optim_name != "adamw_torch":
    raise ValueError("Strict determinism requires --optim_name adamw_torch (non-fused optimizer).")
if val_size_arg is not None and val_size_arg < 1:
    raise ValueError("--val_size must be >= 1 when provided.")
if num_epoch < 1:
    raise ValueError("num_epochs must be >= 1.")
if lr <= 0:
    raise ValueError("learn_rate must be > 0.")
if not (0 <= val_ratio < 1):
    raise ValueError("--val_ratio must be in [0, 1).")
if evals_per_epoch < 1:
    raise ValueError("--evals_per_epoch must be >= 1.")


import numpy as np
import torch


def set_global_seed(seed: int, strict: bool = True) -> None:
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
    torch.use_deterministic_algorithms(True, warn_only=not strict)


def get_pkg_version(pkg_name: str) -> str:
    try:
        return version(pkg_name)
    except Exception:
        return "unknown"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


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


set_global_seed(seed, strict=strict_determinism)
print(f"[ReproConfig] mode={repro_mode} load_in_4bit={load_in_4bit} optim={optim_name} seed={seed}")
print(
    f"[TrainConfig] model_id={model_id} output_dir={output_dir} "
    f"num_epochs={num_epoch} learning_rate={lr} input_json={train_json}"
)
system_message = "You are a Vision-Language Model specialized in univariate time-series anomaly detection. Given a chart image and a prompt, identify whether an anomaly exists, indicate where it occurs, and briefly explain why it is anomalous. Keep your response clear, structured, and focused on visual evidence in the chart."


def format_data(sample):
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
                    {
                        "type": "image",
                        "image": sample["image"],
                    },
                    {
                        "type": "text",
                        "text": sample["query"],
                    },
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": sample["resp"]}],
            },
        ],
    }


train_json_path = Path(train_json).expanduser()
with train_json_path.open("r", encoding="utf-8") as f:
    full_data = json.load(f)

validate_image_paths(full_data, train_json_path)


sample_items = sorted(full_data.items(), key=lambda x: x[0])

if len(sample_items) < 2:
    raise ValueError("Need at least 2 samples to create train/val split.")

if val_size_arg is not None:
    val_size = val_size_arg
else:
    val_size = int(len(sample_items) * val_ratio)


val_size = max(1, min(val_size, len(sample_items) - 1))

random.shuffle(sample_items)
val_items = sample_items[:val_size]
train_items = sample_items[val_size:]

split_metadata = {
    "split_source": "seeded_random",
    "val_size_requested": val_size_arg,
    "val_ratio_requested": val_ratio,
}

print(f"Total samples: {len(sample_items)}")
print(f"Train: {len(train_items)}, Val: {len(val_items)}")

train_dataset = [format_data(sample[1]) for sample in train_items]
eval_dataset = [format_data(sample[1]) for sample in val_items]


from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor


print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

processor = AutoProcessor.from_pretrained(model_id, use_fast=False)


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


if "LOCAL_RANK" in os.environ:
    local_rank = int(os.environ["LOCAL_RANK"])
else:
    local_rank = 0

device = torch.device(f"cuda:{local_rank}")

model_load_kwargs = {
    "torch_dtype": torch.bfloat16,
    "device_map": {"": device},
}

if load_in_4bit:
    from transformers import BitsAndBytesConfig

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model_load_kwargs["quantization_config"] = bnb_config

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_id,
    **model_load_kwargs,
)
model.config.torch_dtype = torch.bfloat16


from trl import SFTConfig

per_device_train_batch_size = 2
per_device_eval_batch_size = 2
gradient_accumulation_steps = 8
world_size = max(1, int(os.environ.get("WORLD_SIZE", "1")))

effective_train_batch = per_device_train_batch_size * gradient_accumulation_steps * world_size
steps_per_epoch = max(1, math.ceil(len(train_dataset) / effective_train_batch))

if eval_strategy_mode == "epoch":
    eval_strategy = "epoch"
    save_strategy = "epoch"
    eval_steps = None
elif eval_strategy_mode == "steps":
    eval_strategy = "steps"
    save_strategy = "steps"
    eval_steps = max(1, steps_per_epoch // evals_per_epoch)
else:
    if steps_per_epoch < 50:
        eval_strategy = "epoch"
        save_strategy = "epoch"
        eval_steps = None
    else:
        eval_strategy = "steps"
        save_strategy = "steps"
        eval_steps = max(1, steps_per_epoch // evals_per_epoch)

print(
    f"[EvalSchedule] mode={eval_strategy_mode} resolved_eval_strategy={eval_strategy} "
    f"steps_per_epoch={steps_per_epoch} eval_steps={eval_steps}"
)

output_dir_path = Path(output_dir)
output_dir_path.mkdir(parents=True, exist_ok=True)
manifest_path = output_dir_path / "training_manifest.json"

manifest = {
    "created_at_utc": utc_now_iso(),
    "argv": sys.argv,
    "python": {
        "version": sys.version,
        "platform": platform.platform(),
    },
    "packages": {
        "torch": get_pkg_version("torch"),
        "transformers": get_pkg_version("transformers"),
        "trl": get_pkg_version("trl"),
        "peft": get_pkg_version("peft"),
        "accelerate": get_pkg_version("accelerate"),
        "bitsandbytes": get_pkg_version("bitsandbytes"),
        "numpy": get_pkg_version("numpy"),
    },
    "env": {
        "WORLD_SIZE": os.environ.get("WORLD_SIZE"),
        "RANK": os.environ.get("RANK"),
        "LOCAL_RANK": os.environ.get("LOCAL_RANK"),
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    },
    "run_config": {
        "model_id": model_id,
        "output_dir": output_dir,
        "num_epochs": num_epoch,
        "learning_rate": lr,
        "seed": seed,
        "repro_mode": repro_mode,
        "optim_name": optim_name,
        "load_in_4bit": load_in_4bit,
        "val_ratio": val_ratio,
        "val_size": val_size,
        "eval_strategy_mode": eval_strategy_mode,
        "evals_per_epoch": evals_per_epoch,
        "resolved_eval_strategy": eval_strategy,
        "resolved_eval_steps": eval_steps,
    },
    "split": {
        **split_metadata,
        "train_count": len(train_items),
        "val_count": len(val_items),
        "train_ids_preview": [sid for sid, _ in train_items[:5]],
        "val_ids_preview": [sid for sid, _ in val_items[:5]],
    },
}
if torch.cuda.is_available():
    manifest["hardware"] = {
        "cuda_available": True,
        "device_count": torch.cuda.device_count(),
        "device_0_name": torch.cuda.get_device_name(0),
    }
else:
    manifest["hardware"] = {"cuda_available": False}

with manifest_path.open("w", encoding="utf-8") as f:
    json.dump(manifest, f, indent=2, ensure_ascii=False)
print(f"[Manifest] Wrote initial manifest to {manifest_path}")

eval_save_kwargs = {
    "eval_strategy": eval_strategy,
    "save_strategy": save_strategy,
}
if eval_strategy == "steps":
    eval_save_kwargs["eval_steps"] = eval_steps
    eval_save_kwargs["save_steps"] = eval_steps

training_args = SFTConfig(
    output_dir=output_dir,
    num_train_epochs=num_epoch,
    per_device_train_batch_size=per_device_train_batch_size,
    per_device_eval_batch_size=per_device_eval_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    max_length=None,
    optim=optim_name,
    learning_rate=lr,
    lr_scheduler_type="cosine",
    weight_decay=0.01,
    logging_steps=10,
    **eval_save_kwargs,
    bf16=True,
    fp16=False,
    max_grad_norm=0.5,
    warmup_ratio=0.03,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    seed=seed,
    data_seed=seed,
    full_determinism=strict_determinism,
    push_to_hub=False,
    hub_token=None,
    report_to=[],
    overwrite_output_dir=True,
    dataloader_num_workers=0,
)


from peft import LoraConfig


peft_config = LoraConfig(
    r=32,
    lora_alpha=64,
    lora_dropout=0.05,
    bias="none",
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    task_type="CAUSAL_LM",
)


from trl import SFTTrainer

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    peft_config=peft_config,
    processing_class=processor,
)

train_result = trainer.train()
trainer.save_model(training_args.output_dir)

manifest["finished_at_utc"] = utc_now_iso()
manifest["train_result"] = getattr(train_result, "metrics", {})
manifest["best_model_checkpoint"] = getattr(trainer.state, "best_model_checkpoint", None)
manifest["best_metric"] = getattr(trainer.state, "best_metric", None)
manifest["global_step"] = getattr(trainer.state, "global_step", None)
manifest["epoch"] = getattr(trainer.state, "epoch", None)
with manifest_path.open("w", encoding="utf-8") as f:
    json.dump(manifest, f, indent=2, ensure_ascii=False)
print(f"[Manifest] Updated final manifest at {manifest_path}")

clear_memory()
