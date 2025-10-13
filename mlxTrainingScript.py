# -*- coding: utf-8 -*-
import json, os
import types
import numpy as np
import matplotlib.pyplot as plt

import mlx.optimizers as optim
from mlx.utils import tree_flatten
from mlx_lm import generate, load
from mlx_lm.tuner import TrainingArgs, train

from Dataset_Gen.utils import build_prompt
from mlx_lm.tuner import linear_to_lora_layers
from mlx_lm.tuner.datasets import load_dataset, CacheDataset
from classes.metrics import LRSchedulerCallback, SimpleMetrics

def main(config=None):
    # Training configuration
    if config == None:
        config = {
            "model_path": "./finetuned_model/models/Qwen3-4B-MLX-bf16_start_end",
            "adapter_dir": "./finetuned_model/adapter/adapters_merged_2", 
            "ds_dir": "data/custom_curriculum_1",
            "max_seq_len": 512,
            "base_lr": 4.0e-5,
            "iters": 10000,
            "warmup": 300,  # 0.03 * iters
            "decay_steps": 9700,  # iters - warmup
            "lr_floor": 4.0e-6,  # 0.1 * base_lr
            "eval_every": 500,
            "training_continue": False,
            "lora_config": {
                "num_layers": 16,
                "lora_parameters": {
                    "rank": 32,
                    "scale": 32.0,
                    "dropout": 0.02,
                }
            }
        }

    os.makedirs(config["adapter_dir"], exist_ok=True)
    adapter_config_path = os.path.join(config["adapter_dir"], "adapter_config.json")
    adapter_file_path   = os.path.join(config["adapter_dir"], "adapters.safetensors")

    # Load model + tokenizer
    if config["training_continue"]:
        model, tokenizer = load(config["model_path"], adapter_path=config["adapter_dir"], tokenizer_config={"trust_remote_code": True})
    else:
        model, tokenizer = load(config["model_path"], tokenizer_config={"trust_remote_code": True})

    # Write LoRA adapter config
    with open(adapter_config_path, "w", encoding="utf-8") as f:
        json.dump(config["lora_config"], f, indent=2)

    # Prepare LoRA
    model.freeze()

    if not config["training_continue"]:
        linear_to_lora_layers(model, config["lora_config"]["num_layers"], config["lora_config"]["lora_parameters"])

    num_train_params = sum(v.size for _, v in tree_flatten(model.trainable_parameters()))
    print(f"Trainable params (LoRA): {num_train_params:,}")

    model.train()

    # Datasets
    ds_args = types.SimpleNamespace(
        data=config["ds_dir"],
        train=True,
        test=True,
        max_seq_len=config["max_seq_len"],
        mask_prompt=True
    )
    train_set, val_set, test_set = load_dataset(ds_args, tokenizer)
    print(f"Loaded train: {len(train_set)}, val: {len(val_set)}, test: {len(test_set)}")

    # Training args & run
    training_args = TrainingArgs(
        adapter_file=adapter_file_path,
        iters=config["iters"],
        steps_per_eval=config["eval_every"],
    )

    cos = optim.cosine_decay(config["base_lr"], config["iters"], config["lr_floor"])
    optimizer = optim.Adam(learning_rate=cos)
    metrics = SimpleMetrics()

    train(
        model=model,
        args=training_args,
        optimizer=optimizer,
        train_dataset=CacheDataset(train_set),
        val_dataset=CacheDataset(val_set),
        training_callback=metrics,
    )

    # Plot losses
    # Create plot directory if it doesn't exist
    plot_dir = f'./plots/{config["adapter_dir"].split("/")[-1]}'
    os.makedirs(plot_dir, exist_ok=True)

    # Save loss data to CSV
    if metrics.train_losses:
        train_its, train_losses = zip(*metrics.train_losses)
        # Save training losses
        with open(f'{plot_dir}/train_losses.csv', 'w') as f:
            f.write('iteration,loss\n')
            for it, loss in zip(train_its, train_losses):
                f.write(f'{it},{loss}\n')
        
        # Plot with log scale for better visualization
        plt.figure(figsize=(10, 6))
        plt.plot(train_its, train_losses, "-o", label="Train", markersize=3)
        
        if metrics.val_losses:
            val_its, val_losses = zip(*metrics.val_losses)
            plt.plot(val_its, val_losses, "-o", label="Validation", markersize=3)
        
        plt.xlabel("Iteration")
        plt.ylabel("Loss (log scale)")
        plt.yscale('log')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{plot_dir}/learning_curve_log.png')
        plt.close()
        
        # Also create a linear plot skipping initial values
        skip_first = max(1, len(train_its) // 10)
        plt.figure(figsize=(10, 6))
        plt.plot(train_its[skip_first:], train_losses[skip_first:], "-o", label="Train", markersize=3)
        
        if metrics.val_losses:
            val_its, val_losses = zip(*metrics.val_losses)
            skip_val = max(1, len(val_its) // 10)
            plt.plot(val_its[skip_val:], val_losses[skip_val:], "-o", label="Validation", markersize=3)
        
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{plot_dir}/learning_curve_zoomed.png')
        plt.close()

if __name__ == "__main__":
    main()
