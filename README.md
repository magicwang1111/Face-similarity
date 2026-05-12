# Face LoRA Eval

Batch face similarity evaluation for selecting the most identity-consistent LoRA checkpoint.

The default config evaluates:

- Reference images: `/mnt/d/测试素材/0417-人脸采集小韩`
- LoRA outputs: `/mnt/e/批量测试输出/20260507人脸替换测试0506rank16模型`
- Local InsightFace models copied into this project: `/mnt/e/Face-similarity/models/insightface`
- Model: `buffalo_l`
- Providers: `CUDAExecutionProvider` first, then `CPUExecutionProvider` fallback

## Install

```bash
cd /mnt/e/Face-similarity
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e ".[test]"
```

If `python3 -m venv` or `pip` is unavailable on the machine, install Python's venv/pip package first, then run the commands above.

## Run

```bash
python -m face_lora_eval evaluate --config configs/xiaohan_20260507.yaml
```

On Windows, the tested ComfyUI conda environment command is:

```bat
cd /d E:\Face-similarity
set PYTHONPATH=src
D:\miniconda3\Scripts\conda.exe run -n comfyui python -m face_lora_eval evaluate --config configs\xiaohan_20260507.yaml
```

Useful smoke test:

```bash
python -m face_lora_eval evaluate \
  --config configs/xiaohan_20260507.yaml \
  --limit-reference 1 \
  --limit-candidates-per-lora 2
```

Reports are written to:

```text
/mnt/e/Face-similarity/reports/xiaohan_20260507/
```

The project-local model directory on this machine contains both `buffalo_l` and `antelopev2`:

```text
/mnt/e/Face-similarity/models/insightface/models/buffalo_l
/mnt/e/Face-similarity/models/insightface/models/antelopev2
```

The ONNX model files are ignored by git because they are hundreds of MB. See `models/insightface/README.md` for the restore/copy command.

## GPU Note

The code requests `CUDAExecutionProvider` first and falls back to CPU. In the tested `comfyui` conda environment, PyTorch CUDA sees the RTX 4090 D, but `onnxruntime` currently exposes only CPU/Azure providers. Install a matching `onnxruntime-gpu` build in that environment to enable GPU inference.

## Privacy Boundary

Use this only for authorized LoRA quality evaluation. Images and embeddings stay local. Do not use this workflow for unauthorized identity recognition, surveillance, tracking, or large-scale face search.
