#!/usr/bin/env bash

#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --job-name=cpcfg_text
#SBATCH --gpus=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G

source $(conda info --base)/etc/profile.d/conda.sh
conda activate myvcpcfgenv

language=$1

python -m cli train --config configs/joint_cpcfg_default.ini \
    model=text_cpcfg_${language} \
    train_sents=data/abstractScenes_${language}.senttoks \
    valid_sents=data/abstractScenes_${language}.senttoks \
    valid_trees=data/abstractScenes_${language}.senttrees \
    joint_training=false \
    vse_mt_alpha=0.0