# Repository Guidelines

## Project Structure & Module Organization
- `train_unziplora.py`: main training entry (Accelerate + SDXL + dual LoRA optimization).
- `infer.py`: inference entry for content/style/merged generation.
- `unziplora_unet/`: core modified UNet and attention stack (`attention_processor.py`, `unziplora_linear_layer.py`, `utils.py`, etc.).
- `record_utils/`: analysis helpers (e.g., cone/heatmap-related utilities).
- `instance_data/`: example training images and prompt files.
- `models/`: saved checkpoints, LoRA weights, and merger weights.
- `train.sh`, `infer.sh`: canonical runnable examples for local workflows.

## Build, Test, and Development Commands
- Install environment:
  - `conda create -n unziplora python=3.11 && conda activate unziplora`
  - `pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu118`
  - `pip install -r requirements.txt`
- Train:
  - `bash train.sh`
  - or `accelerate launch train_unziplora.py ...` for custom experiments.
- Infer:
  - `bash infer.sh`
  - or `accelerate launch infer.py ...` with explicit prompts/output paths.

## Coding Style & Naming Conventions
- Python style: 4-space indentation, follow existing PyTorch/diffusers-oriented patterns.
- Use `snake_case` for functions/variables, `PascalCase` for classes (e.g., `UnZipLoRALinearLayer`).
- Keep tensor-shape comments concise near non-obvious operations.
- No enforced formatter/linter is configured in-repo; keep changes consistent with nearby code and avoid broad reformatting.

## Testing Guidelines
- There is currently no dedicated `tests/` suite or pytest setup.
- For validation, run a short smoke workflow:
  - 1) small-step training via `train.sh` (reduced `--max_train_steps`)
  - 2) inference via `infer.sh`
  - 3) verify outputs/checkpoints under `models/` and generated images.
- When changing `unziplora_unet/`, include a minimal reproduction command in your PR notes.

## Commit & Pull Request Guidelines
- Recent history uses short, focused commit messages (often Chinese), describing the concrete change.
- Prefer one logical change per commit; include touched area first when possible, e.g.:
  - `train: fix focus-mask alignment in both forward`
- PRs should include:
  - purpose and scope,
  - key flags used (`--focus`, `--with_align_loss`, etc.),
  - expected impact on training/inference,
  - sample outputs or logs (screenshots/W&B links) for behavior changes.
