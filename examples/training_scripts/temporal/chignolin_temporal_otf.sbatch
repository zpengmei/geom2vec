#!/bin/bash

#SBATCH --job-name=vis_otf_temporal_chignolin
#SBATCH --output=%x.out
#SBATCH --error=%x.err

#SBATCH --account=pi-dinner

#SBATCH --time=2-00:00:00
#SBATCH --nodes=1

#SBATCH --partition=beagle3
#SBATCH --ntasks=8
#SBATCH --mem=36G
#SBATCH --gres=gpu:2

SEED=(1 2 3)
SYSTEM=chignolin
TRAINING_FRACTIONS=(0.1 0.2 0.3 0.5 0.8)

for seed in "${SEED[@]}"; do
  system=${SYSTEM}
  for fraction in "${TRAINING_FRACTIONS[@]}"; do
    save_dir="/project/dinner/zpengmei/geom2vec/runs/temporal/otf/${system}/${fraction}/${seed}"
    mkdir -p $save_dir
    python /project/dinner/zpengmei/geom2vec/examples/training_scripts/temporal/training_otf.py \
      --training_fraction ${fraction} \
      --vector_feature \
      --save_dir ${save_dir} \
      --seed ${seed} \
      --system ${system} \
      --batch_size 1000 \
      --n_epochs 50 \
      --train_patience 200 \
      --train_valid_interval 200 \
      --valid_patience 5
  done
done



