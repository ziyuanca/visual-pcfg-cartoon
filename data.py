"""Data loading and batching"""

from __future__ import annotations

import gzip
import logging
import os
import random
from collections import Counter
from dataclasses import dataclass
from typing import List, Mapping, Optional, Sequence, Tuple

import nltk
import numpy as np
import torch

from korean_phonetic_vocab import get_korean_phone_mappings, translate_phone_to_ids

EOS = "<eos>"
BOS = "<bos>"
PAD = "<pad>"
OOV = "<oov>"
BOW = "<bow>"
EOW = "<eow>"
LRB = "-LRB-"
RRB = "-RRB-"

SPECIAL_WORDS = (OOV, BOS, EOS, PAD, LRB, RRB)
SPECIAL_CHARS = (BOS, EOS, OOV, PAD, BOW, EOW)
MAX_TOKEN_LENGTH = 28


Sentence = List[str]
Dataset = List[Sentence]


@dataclass(frozen=True)
class Vocabularies:
    """Word and character id mappings built from the training split."""

    word_to_id: Mapping[str, int]
    char_to_id: Mapping[str, int]


@dataclass(frozen=True)
class ImageAlignment:
    """Global image table and one image id per sentence."""

    embeddings: torch.Tensor
    sentence_image_ids: torch.Tensor


@dataclass(frozen=True)
class Batch:
    """One model batch with explicit fields instead of tuple positions."""

    word_ids: Optional[torch.Tensor]
    char_ids: Optional[torch.Tensor]
    variable_chars: Optional[List[List[torch.Tensor]]]
    lengths: List[int]
    sentence_indices: List[int]
    images: Optional[torch.Tensor] = None


@dataclass(frozen=True)
class DataPipelineConfig:
    """Configuration needed to prepare data for train/eval."""

    train_sents: str
    valid_sents: Optional[str] = None
    valid_trees: Optional[str] = None
    train_image_embs: Optional[str] = None
    train_image_ids: Optional[str] = None
    batch_size: int = 2
    max_vocab_size: int = 150000
    min_count: int = 1
    device: str = "cpu"
    eval_device: str = "cpu"
    parser_type: str = "simple"
    group_by_length: bool = False
    shuffle: bool = True
    sort: bool = True


@dataclass(frozen=True)
class PreparedData:
    """Fully prepared data objects for model construction and training."""

    train_sentences: Dataset
    valid_sentences: Optional[Dataset]
    valid_trees: Optional[List[str]]
    vocabularies: Vocabularies
    train_batches: List[Batch]
    valid_batches: Optional[List[Batch]]
    train_images: Optional[ImageAlignment] = None


def _truncate_token(token: str) -> str:
    if len(token) > MAX_TOKEN_LENGTH:
        return token[:14] + token[-14:]
    return token


def read_corpus(
    path: os.PathLike | str,
    korean_phonetics: bool = False,
) -> Dataset:
    """Read a sentence-token file using the current pipeline's normalization.

    Each non-empty line becomes one sentence, lowercased and wrapped in
    ``<bos>/<eos>``.
    """

    dataset: Dataset = []
    longest_word = 0
    korean_mapping = None
    if korean_phonetics:
        korean_mapping = get_korean_phone_mappings()

    path_str = os.fspath(path)
    open_text = gzip.open if path_str.endswith(".gz") else open
    with open_text(path_str, "rt", encoding="utf-8") as fin:
        for line_no, line in enumerate(fin, start=1):
            raw_tokens = line.strip().split()
            if not raw_tokens:
                raise ValueError(f"Empty sentence at line {line_no}: {os.fspath(path)}")

            sent = [BOS]
            for token in raw_tokens:
                token = _truncate_token(token)
                if korean_phonetics:
                    token = translate_phone_to_ids(token, korean_mapping)
                token = token.lower()
                longest_word = max(longest_word, len(token))
                sent.append(token)
            sent.append(EOS)
            longest_word = max(longest_word, len(BOS), len(EOS))
            dataset.append(sent)

    if dataset:
        logging.info("Longest word in the data:%s", longest_word)
    else:
        logging.info("Longest word in the data:0")
    return dataset


def load_gold_trees(
    path: os.PathLike | str,
    sentences: Optional[Dataset] = None,
    validate_leaf_count: bool = True,
) -> List[str]:
    """Load and optionally validate gold constituency trees."""

    with open(path, "r", encoding="utf-8") as fh:
        trees = [line.strip() for line in fh]

    if sentences is None:
        return trees

    if len(trees) != len(sentences):
        raise ValueError(
            "Gold tree count mismatch: "
            f"{len(trees)} trees but {len(sentences)} validation sentences."
        )


    for index, (tree_str, sent) in enumerate(zip(trees, sentences)):
        tree = nltk.Tree.fromstring(tree_str)
        if validate_leaf_count:
            expected = max(len(sent) - 2, 0)
            actual = len(tree.leaves())
            if actual != expected:
                raise ValueError(
                    "Gold tree leaf mismatch at sentence {}: "
                    "tree has {} leaves but sentence has {} content tokens.".format(
                        index, actual, expected
                    )
                )
    return trees


def get_truncated_vocab(
    dataset: Dataset,
    min_count: int,
    max_num: int,
) -> List[Tuple[str, int]]:
    """Return the frequency-sorted vocabulary slice used by the pipeline."""

    word_count = Counter()
    for sentence in dataset:
        word_count.update(sentence)

    print(word_count.most_common(10))

    sorted_counts = list(word_count.items())
    sorted_counts.sort(key=lambda x: x[1], reverse=True)

    cutoff = 0
    for _, count in sorted_counts:
        if count < min_count:
            break
        cutoff += 1
    cutoff = min(cutoff, max_num)

    logging.info(
        "Truncated word count: %s.",
        sum(count for _, count in sorted_counts[cutoff:]),
    )
    logging.info(
        "Original vocabulary size: %s. Truncated vocab size %s.",
        len(sorted_counts),
        cutoff,
    )
    return sorted_counts[:cutoff]


def build_vocabularies(
    train_sentences: Dataset,
    max_vocab_size: int = 150000,
    min_count: int = 1,
) -> Vocabularies:
    """Build word/char vocabularies from train sentences only."""

    word_to_id: dict[str, int] = {}
    for token in SPECIAL_WORDS:
        word_to_id[token] = len(word_to_id)

    vocab = get_truncated_vocab(
        train_sentences,
        min_count=min_count,
        max_num=max_vocab_size,
    )
    for word, _ in vocab:
        if word not in word_to_id:
            word_to_id[word] = len(word_to_id)

    char_to_id: dict[str, int] = {}
    for sentence in train_sentences:
        for word in sentence:
            for ch in word:
                if ch not in char_to_id:
                    char_to_id[ch] = len(char_to_id)

    for token in SPECIAL_CHARS:
        if token not in char_to_id:
            char_to_id[token] = len(char_to_id)

    logging.info(
        "Vocabulary size: %s; Max length: %s",
        len(word_to_id),
        max((len(x) for x in word_to_id), default=0),
    )
    logging.info("Char embedding size: %s", len(char_to_id))
    return Vocabularies(word_to_id=word_to_id, char_to_id=char_to_id)


def write_vocabularies(vocabularies: Vocabularies, output_dir: os.PathLike | str) -> None:
    """Write ``word.dic`` and ``char.dic`` in the current artifact format."""

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "char.dic"), "w", encoding="utf-8") as fpo:
        for ch, i in vocabularies.char_to_id.items():
            print(f"{ch}\t{i}", file=fpo)

    with open(os.path.join(output_dir, "word.dic"), "w", encoding="utf-8") as fpo:
        for word, i in vocabularies.word_to_id.items():
            print(f"{word}\t{i}", file=fpo)


def _normalize_optional_path(path: Optional[os.PathLike | str]) -> str:
    if path is None:
        return ""
    return os.fspath(path).strip()


def load_image_embeddings_with_sentence_ids(
    dataset: Dataset,
    image_embeddings_path: os.PathLike | str,
    sentence_image_ids_path: os.PathLike | str,
) -> ImageAlignment:
    """Load image embeddings."""

    image_embeddings = np.load(image_embeddings_path)
    if image_embeddings.ndim != 2:
        raise ValueError(
            "Image embeddings must be a 2D array. "
            f"Found shape {image_embeddings.shape} from {image_embeddings_path}."
        )

    sentence_image_ids = []
    with open(sentence_image_ids_path, "r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            raw = line.strip()
            sentence_image_ids.append(int(raw))

    if len(sentence_image_ids) != len(dataset):
        raise ValueError(
            "Sentence/image alignment mismatch: "
            f"{len(dataset)} sentences but {len(sentence_image_ids)} ids in "
            f"{sentence_image_ids_path}."
        )

    if sentence_image_ids:
        min_id = min(sentence_image_ids)
        max_id = max(sentence_image_ids)
        if min_id < 0 or max_id >= image_embeddings.shape[0]:
            raise ValueError(
                "Image id out of range for embedding table. "
                f"Valid range: [0, {image_embeddings.shape[0] - 1}], "
                f"found min={min_id}, max={max_id} in {sentence_image_ids_path}."
            )

    return ImageAlignment(
        embeddings=torch.as_tensor(image_embeddings, dtype=torch.float32),
        sentence_image_ids=torch.as_tensor(sentence_image_ids, dtype=torch.long),
    )


def _validate_image_pair(
    split: str,
    image_embeddings_path: Optional[os.PathLike | str],
    sentence_image_ids_path: Optional[os.PathLike | str],
) -> Tuple[str, str]:
    image_embeddings_path = _normalize_optional_path(image_embeddings_path)
    sentence_image_ids_path = _normalize_optional_path(sentence_image_ids_path)
    if bool(image_embeddings_path) != bool(sentence_image_ids_path):
        raise ValueError(
            f"Both {split}_image_embs and {split}_image_ids must be set together."
        )
    return image_embeddings_path, sentence_image_ids_path


def _lookup_id(mapping: Mapping[str, int], key: str, name: str) -> int:
    value = mapping.get(key)
    if value is None:
        raise ValueError(f"{name} vocabulary is missing required token {key!r}.")
    return value


def _make_char_tensor(
    token: str,
    char_to_id: Mapping[str, int],
    bow_id: int,
    eow_id: int,
    oov_id: int,
    device: str,
) -> torch.Tensor:
    ids = [bow_id]
    ids.extend(char_to_id.get(ch, oov_id) for ch in token)
    ids.append(eow_id)
    return torch.tensor(ids, dtype=torch.long, device=device)


def _build_batch(
    sentences: Sequence[Sentence],
    sentence_indices: Sequence[int],
    word_to_id: Optional[Mapping[str, int]],
    char_to_id: Optional[Mapping[str, int]],
    sort: bool,
    device: str,
    image_ids: Optional[Sequence[int]] = None,
    image_table: Optional[torch.Tensor] = None,
) -> Batch:
    batch_size = len(sentences)
    if batch_size == 0:
        raise ValueError("Cannot build an empty batch.")

    local_order = list(range(batch_size))
    if sort:
        local_order.sort(key=lambda idx: -len(sentences[idx]))

    ordered_sentences = [list(sentences[i]) for i in local_order]
    ordered_indices = [int(sentence_indices[i]) for i in local_order]
    ordered_image_ids = None
    if image_ids is not None:
        if len(image_ids) != batch_size:
            raise ValueError(
                f"image_ids length mismatch in batch: expected {batch_size}, "
                f"found {len(image_ids)}"
            )
        ordered_image_ids = [int(image_ids[i]) for i in local_order]

    lengths = [len(sent) for sent in ordered_sentences]
    max_len = max(lengths)

    if word_to_id is None:
        word_ids = None
    else:
        oov_id = _lookup_id(word_to_id, OOV, "word")
        pad_id = _lookup_id(word_to_id, PAD, "word")
        word_ids = torch.full(
            (batch_size, max_len),
            pad_id,
            dtype=torch.long,
            device=device,
        )
        for sent_idx, sent in enumerate(ordered_sentences):
            for token_idx, token in enumerate(sent):
                word_ids[sent_idx, token_idx] = word_to_id.get(token, oov_id)

    if char_to_id is None:
        char_ids = None
        variable_chars = None
    else:
        bow_id = _lookup_id(char_to_id, BOW, "char")
        eow_id = _lookup_id(char_to_id, EOW, "char")
        oov_id = _lookup_id(char_to_id, OOV, "char")
        pad_id = _lookup_id(char_to_id, PAD, "char")
        max_chars = max(len(token) for sent in ordered_sentences for token in sent) + 2
        char_ids = torch.full(
            (batch_size, max_len, max_chars),
            pad_id,
            dtype=torch.long,
            device=device,
        )
        variable_chars = []
        for sent_idx, sent in enumerate(ordered_sentences):
            variable_chars.append([])
            for token_idx, token in enumerate(sent):
                char_ids[sent_idx, token_idx, 0] = bow_id
                if token in (BOS, EOS):
                    char_ids[sent_idx, token_idx, 1] = char_to_id[token]
                    char_ids[sent_idx, token_idx, 2] = eow_id
                    continue

                variable_chars[sent_idx].append(
                    _make_char_tensor(token, char_to_id, bow_id, eow_id, oov_id, device)
                )
                for char_idx, ch in enumerate(token):
                    char_ids[sent_idx, token_idx, char_idx + 1] = char_to_id.get(ch, oov_id)
                char_ids[sent_idx, token_idx, len(token) + 1] = eow_id

    images = None
    if image_table is not None:
        if ordered_image_ids is None:
            raise ValueError("image_table was provided without image ids.")
        image_id_tensor = torch.as_tensor(ordered_image_ids, dtype=torch.long)
        images = image_table[image_id_tensor].to(device, non_blocking=True)
    elif ordered_image_ids is not None:
        raise ValueError("image ids were provided without image_table.")

    return Batch(
        word_ids=word_ids,
        char_ids=char_ids,
        variable_chars=variable_chars,
        lengths=lengths,
        sentence_indices=ordered_indices,
        images=images,
    )


def create_batches(
    sentences: Dataset,
    batch_size: int,
    word_to_id: Optional[Mapping[str, int]],
    char_to_id: Optional[Mapping[str, int]],
    eval: bool = False,
    shuffle: bool = True,
    sort: bool = True,
    device: str = "cpu",
    eval_device: str = "cpu",
    all_image_embs: Optional[torch.Tensor] = None,
    sentence_image_ids: Optional[Sequence[int] | torch.Tensor] = None,
    group_by_length: bool = False,
) -> List[Batch]:
    """Create named batches."""

    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, found {batch_size}.")
    if not sentences:
        raise ValueError("Cannot create batches from an empty dataset.")

    target_device = eval_device if eval else device

    order = list(range(len(sentences)))
    if shuffle:
        random.shuffle(order)

    if sort and group_by_length:
        order.sort(key=lambda idx: -len(sentences[idx]))

    if (all_image_embs is None) != (sentence_image_ids is None):
        raise ValueError("all_image_embs and sentence_image_ids must be provided together.")

    sorted_sentences = [sentences[i] for i in order]
    sorted_sentence_image_ids = None
    image_table = None
    if sentence_image_ids is not None:
        if len(sentence_image_ids) != len(sentences):
            raise ValueError(
                "sentence_image_ids length mismatch: "
                f"expected {len(sentences)}, found {len(sentence_image_ids)}."
            )
        if torch.is_tensor(sentence_image_ids):
            sentence_image_ids = sentence_image_ids.tolist()
        sorted_sentence_image_ids = [int(sentence_image_ids[i]) for i in order]
        image_table = torch.as_tensor(all_image_embs, dtype=torch.float32)

    batch_positions: List[List[int]] = []
    if group_by_length:
        start_id = 0
        while start_id < len(sorted_sentences):
            end_id = start_id + 1
            current_len = len(sorted_sentences[start_id])
            while (
                end_id < len(sorted_sentences)
                and len(sorted_sentences[end_id]) == current_len
            ):
                end_id += 1

            positions = list(range(start_id, end_id))
            for batch_start in range(0, len(positions), batch_size):
                batch_positions.append(positions[batch_start : batch_start + batch_size])
            start_id = end_id
    else:
        positions = list(range(len(sorted_sentences)))
        for batch_start in range(0, len(positions), batch_size):
            batch_positions.append(positions[batch_start : batch_start + batch_size])

    batches: List[Batch] = []
    sum_len = 0.0
    for positions in batch_positions:
        batch_sentences = [sorted_sentences[pos] for pos in positions]
        sentence_indices = [order[pos] for pos in positions]
        image_ids = None
        if sorted_sentence_image_ids is not None:
            image_ids = [sorted_sentence_image_ids[pos] for pos in positions]

        batch = _build_batch(
            batch_sentences,
            sentence_indices,
            word_to_id,
            char_to_id,
            sort=sort,
            device=target_device,
            image_ids=image_ids,
            image_table=image_table,
        )
        sum_len += sum(batch.lengths)
        batches.append(batch)

    all_lens = [len(sent) for sent in sorted_sentences]
    logging.info(
        "%s batches, avg len: %.1f, max len %s, min len %s.",
        len(batches),
        sum_len / len(sentences),
        max(all_lens),
        min(all_lens),
    )

    if shuffle and not eval:
        batch_order = list(range(len(batches)))
        random.shuffle(batch_order)
        return [batches[i] for i in batch_order]
    return batches


def _load_optional_images(
    split: str,
    sentences: Optional[Dataset],
    embeddings_path: Optional[os.PathLike | str],
    ids_path: Optional[os.PathLike | str],
) -> Optional[ImageAlignment]:
    embeddings_path, ids_path = _validate_image_pair(split, embeddings_path, ids_path)
    if not embeddings_path:
        return None
    if sentences is None:
        raise ValueError(f"{split}_image_* provided but {split}_sents is not configured.")
    return load_image_embeddings_with_sentence_ids(
        sentences,
        embeddings_path,
        ids_path,
    )


def prepare_data(config: DataPipelineConfig) -> PreparedData:
    """Prepare train/valid sentences, vocabularies, train images, and batches."""

    train_sentences = read_corpus(config.train_sents)
    valid_sentences = (
        None
        if _normalize_optional_path(config.valid_sents) == ""
        else read_corpus(config.valid_sents)
    )
    valid_trees = None
    if _normalize_optional_path(config.valid_trees):
        if valid_sentences is None:
            raise ValueError("valid_trees provided but valid_sents is not configured.")
        valid_trees = load_gold_trees(config.valid_trees, valid_sentences)

    train_images = _load_optional_images(
        "train",
        train_sentences,
        config.train_image_embs,
        config.train_image_ids,
    )
    vocabularies = build_vocabularies(
        train_sentences,
        max_vocab_size=config.max_vocab_size,
        min_count=config.min_count,
    )

    train_batches = create_batches(
        train_sentences,
        config.batch_size,
        vocabularies.word_to_id,
        vocabularies.char_to_id,
        device=config.device,
        eval_device=config.eval_device,
        all_image_embs=None if train_images is None else train_images.embeddings,
        sentence_image_ids=None if train_images is None else train_images.sentence_image_ids,
        group_by_length=config.group_by_length,
        shuffle=config.shuffle,
        sort=config.sort,
    )

    valid_batches = None
    if valid_sentences is not None:
        valid_batches = create_batches(
            valid_sentences,
            config.batch_size,
            vocabularies.word_to_id,
            vocabularies.char_to_id,
            eval=True,
            device=config.device,
            eval_device=config.eval_device,
            group_by_length=config.group_by_length,
            shuffle=config.shuffle,
            sort=config.sort,
        )

    return PreparedData(
        train_sentences=train_sentences,
        valid_sentences=valid_sentences,
        valid_trees=valid_trees,
        vocabularies=vocabularies,
        train_batches=train_batches,
        valid_batches=valid_batches,
        train_images=train_images,
    )
