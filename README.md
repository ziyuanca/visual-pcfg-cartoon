# Visual-pcfg-cartoon

Adapted from [Portelance et al. (2025)](https://github.com/evaportelance/structure-meaning-learning) and [Jin et al. (2021)](https://github.com/lifengjin/charInduction).


## Data
[data](data/) includes sentences and silver-standard trees for 6 languages: English(en), German(de), Hebrew(he), Hungarian(hu), Korean(ko), and Chinese(zh).

The English sentences and trees come from Portelance et al. (2025). Sentences in other languages are translated from the original English sentences using Google Translate API. The silver-standard trees are produced by the [Berkeley Neural Parser](https://github.com/nikitakit/self-attentive-parser).

## Train

To train a simple PCFG with image matching loss for English, use
```
python -m cli train --config configs/joint_pcfg_default.ini \
    model=joint_pcfg_en \
    train_sents=data/abstractScenes_en.senttoks \
    valid_sents=data/abstractScenes_en.senttoks \
    valid_trees=data/abstractScenes_en.senttrees \
    train_image_embs=data/train_image_embs.npy \
    train_image_ids=data/train_image_ids.id
```

To train a compound PCFG with image matching loss for English, use
```
python -m cli train --config configs/joint_cpcfg_default.ini \
    model=joint_cpcfg_en \
    train_sents=data/abstractScenes_en.senttoks \
    valid_sents=data/abstractScenes_en.senttoks \
    valid_trees=data/abstractScenes_en.senttrees \
    train_image_embs=data/train_image_embs.npy \
    train_image_ids=data/train_image_ids.id
```

To train without the image matching loss, set `joint_training=false`. To use the character-level emission model, set `model_type=char`.