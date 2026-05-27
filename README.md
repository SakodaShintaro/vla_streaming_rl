# vla_streaming_rl

Reinforcement learning experiments with visual observations.

## Installation

NVIDIA GPU with driver is required for training.

### Clone with submodules

The CARLA training env depends on the [Bench2Drive](https://github.com/SakodaShintaro/Bench2Drive) submodule under `external/`.

```bash
# Fresh clone — pull submodules in one shot.
git clone --recursive <repo-url>

# Already cloned without --recursive — fetch submodules now.
git submodule update --init --recursive

# After pulling new commits that bumped the submodule pointer.
git submodule update --recursive
```

### Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Setup project

```bash
uv sync
```

### Login to Hugging Face (for model downloads)

```bash
uv run huggingface-cli login
```

### (Optional) Setup CARLA

```bash
# Download binary from the following URL and extract to $HOME
https://github.com/carla-simulator/carla/releases

# Install dependencies
pip install ~/CARLA_0.9.16/PythonAPI/carla/dist/carla-0.9.16-cp310-cp310-manylinux_2_31_x86_64.whl
```

### (Optional) Setup Animal-AI

Animal-AI v5 does not auto-download the Unity binary, so place it manually.

```bash
# Download Linux.zip from the Animal-AI releases page:
#   https://github.com/Kinds-of-Intelligence-CFI/animal-ai/releases
# (verified with v4.3.0 Linux build)

mkdir -p ~/animalai_env
unzip Linux.zip -d ~/animalai_env/
chmod +x ~/animalai_env/Linux/animalAI.x86_64
```

The training script (`train_animalai.sh`) expects the binary at
`~/animalai_env/Linux/animalAI.x86_64`. The arena (task) is selected via
`configs/env/animalai.yaml` (`env_factory.arena_yaml`); `null` falls back
to the `GoodGoal_Random.yml` bundled inside the `animalai` package.

### Setup pre-commit hooks

Ruff lint (`--fix`) and ruff-format are applied to `*.py` / `*.pyi` files on every `git commit`.

```bash
uv tool install pre-commit
pre-commit install
```

Useful commands:

```bash
# Run hooks against all files (not just staged ones).
pre-commit run --all-files

# Bypass hooks for a single commit.
git commit --no-verify
```

If hooks rewrite files, the commit aborts — re-`git add` the changes and commit again.

## Usage

### Training

```bash
./train_car_racing_on_policy.sh
```

### Testing

```bash
./test.sh
```

## Project Structure

```bash
vla_streaming_rl/
├── src/vla_streaming_rl/     # Library code
│   ├── agents/          # RL agents (on-policy, off-policy)
│   ├── envs/            # Custom environments
│   ├── networks/        # Neural network architectures
│   ├── metrics/         # Metrics computation
│   └── optimizers/      # Custom optimizers
├── scripts/             # Executable scripts
│   └── train.py         # Main training script
└── train_*.sh           # Training shell scripts
```
