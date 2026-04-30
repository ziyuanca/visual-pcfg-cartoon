#!/usr/bin/env bash

#SBATCH --time=024:00:00
#SBATCH --nodes=1
#SBATCH --job-name=cpcfg_joint_char
#SBATCH --gpus=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G

source $(conda info --base)/etc/profile.d/conda.sh
conda activate myvcpcfgenv

language=$1

python -m cli train --config configs/joint_cpcfg_default.ini \
    model=joint_cpcfg_char_${language} \
    train_sents=data/abstractScenes_${language}.senttoks \
    valid_sents=data/abstractScenes_${language}.senttoks \
    valid_trees=data/abstractScenes_${language}.senttrees \
    train_image_embs=data/train_image_embs.npy \
    train_image_ids=data/train_image_ids.id \
    model_type=char