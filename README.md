
### Environment creation

```commandline
python -m venv .venv && source .venv/bin/activate
pip install -U mlx mlx-vlm datasets pillow
pip install opencv-python soundfile
```
(optional) for video/audio VLMs
and have ffmpeg on PATH if you’ll touch video/audio

