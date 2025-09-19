### Environment creation

```commandline
python -m venv .venv && source .venv/bin/activate
pip install -U mlx mlx-vlm datasets pillow
pip install opencv-python soundfile
```
(optional) for video/audio VLMs
and have ffmpeg on PATH if you’ll touch video/audio


to train using lora
```commandline
mlx_lm.lora \
    --model nightmedia/Qwen3-4B-Thinking-2507-bf16-mlx \
    --data data/custom \
    --train \
    --fine-tune-type dora \
    --batch-size 100 \
    --num-layers 16 \
    --iters 500 \
    --adapter-path Fine-tuned_models/adapters/qwen3_maze_dora
```

merge lora adapter as standalone model:
```commandline
mlx_lm.fuse \
    --model nightmedia/Qwen3-4B-Thinking-2507-bf16-mlx \
    --save-path Fine-tuned_models/models/qwen3_maze_thinking \
    --adapter-path Fine-tuned_models/adapters/qwen3_maze_dora
```

