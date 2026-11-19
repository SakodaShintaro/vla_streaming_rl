# PBS

## sif build

```bash
singularity build --fakeroot pbs/container/xvfb.sif pbs/container/xvfb.def
```

## run

`args` holds the script's arguments as one string, taken from the environment so that Hydra
overrides (which contain `=`) pass through untouched.

```bash
args="life siglip trial001 siglip.num_gdn_layers=1" qsub -v script=train_animalai.sh,args pbs/job.sh
```

The container has no `unzip`, so run `./setup_animalai.sh` once on the login node before the
first Animal-AI job.
