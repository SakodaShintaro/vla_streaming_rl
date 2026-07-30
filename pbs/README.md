# PBS

## sif build

```bash
singularity build --fakeroot pbs/container/xvfb.sif pbs/container/xvfb.def
```

## run

```bash
qsub -v script=train_animalai.sh,exp_name=trial001 pbs/job.sh
```
