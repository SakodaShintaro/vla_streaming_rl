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

`./setup.sh` downloads CARLA 0.9.16 and AdditionalMaps into `$HOME/CARLA_0.9.16`
and runs `ImportAssets.sh` (~23 GB total, large download). The Python bindings
are installed via `uv sync` from the wheel referenced in `pyproject.toml`.

### (Optional) Setup Animal-AI

Animal-AI v5 does not auto-download the Unity binary, so fetch it once:

```bash
./setup_animalai.sh
```

That installs the 4.3.2_alpha2 Linux build of
[SakodaShintaro/animal-ai-unity](https://github.com/SakodaShintaro/animal-ai-unity)
into `~/animalai_env/4.3.2_alpha2/`, which is where
`configs/env/animalai.yaml` points `env_factory.binary_path`. It is the official
player rebuilt so it also accepts continuous actions and renders the top-down
camera; in discrete mode it is step-for-step identical to the official 4.3.x
release. `train_animalai.sh` runs the script itself, so this is only needed to
install the binary ahead of time.

Do not measure on 4.3.2_alpha1: the two tunnel prefabs are silently missing from
that build, which leaves the 21 competition arenas built around them as open
floor.

Which arenas an episode draws from is set in `configs/env/animalai.yaml` under
`env_factory` (`mode`, `train_variant`, `train_level`).

### Setup pre-commit hooks

Ruff lint (`--fix`) and ruff-format are applied to `*.py` / `*.pyi` files on every `git commit`.

```bash
uv tool install pre-commit
pre-commit install
```

## Usage

### Training

```bash
./train_car_racing_on_policy.sh
```

### Testing

```bash
./test.sh
```
