"""Evaluation and tree output"""

from __future__ import annotations

import gzip
import logging
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import nltk
import torch

from eval.evalb_unlabeled import eval_rvm_et_al

from data import BOS, EOS, Batch, Sentence


@dataclass(frozen=True)
class ParseResult:
    total_structure_score: float
    average_structure_score: float
    trees: List[object]


@dataclass(frozen=True)
class EvaluationMetrics:
    precision: float
    recall: float
    f1: float
    homogeneity: float
    rh: float
    vm: float


@dataclass(frozen=True)
class EvaluationResult:
    parse_result: ParseResult
    tree_path: Path
    predicted_trees: List[object]
    metrics: Optional[EvaluationMetrics] = None


def parse_batches(
    model,
    batches: Sequence[Batch],
    epoch: int,
    writer=None,
    section: str = "dev",
    logger: Optional[logging.Logger] = None,
) -> ParseResult:
    """Decode Viterbi parses for named batches and restore dataset order."""

    was_training = getattr(model, "training", False)
    model.eval()
    writer = writer if writer is not None else getattr(model, "writer", None)

    total_structure_score = 0.0
    total_num_tags = sum(sum(batch.lengths) for batch in batches)
    trees = _empty_tree_list(batches)

    try:
        with torch.no_grad():
            for batch_index, batch in enumerate(batches):
                structure_score, batch_trees = model.parse(
                    batch.word_ids,
                    batch.variable_chars,
                    batch.sentence_indices,
                    set_grammar=(batch_index == 0),
                    lengths=batch.lengths,
                )
                if len(batch_trees) != len(batch.sentence_indices):
                    raise ValueError(
                        "Parse output size mismatch: "
                        f"{len(batch_trees)} trees for {len(batch.sentence_indices)} sentences."
                    )
                for sentence_index, tree in zip(batch.sentence_indices, batch_trees):
                    trees[sentence_index] = tree
                total_structure_score += float(structure_score)
    finally:
        if was_training:
            model.train()

    average_structure_score = (
        total_structure_score / total_num_tags if total_num_tags > 0 else 0.0
    )
    if writer is not None:
        writer.add_scalar(
            f"{section}_epochwise/average_structure_loss",
            average_structure_score,
            epoch,
        )
    if logger is not None:
        logger.info(
            "Epoch %s EVALUATION | Structure loss %.4f",
            epoch,
            total_structure_score,
        )
    return ParseResult(
        total_structure_score=total_structure_score,
        average_structure_score=average_structure_score,
        trees=trees,
    )


def likelihood_batches(
    model,
    batches: Sequence[Batch],
    epoch: int,
    writer=None,
    section: str = "dev",
    logger: Optional[logging.Logger] = None,
) -> float:
    """Compute total structure likelihood/loss over batches without decoding."""

    was_training = getattr(model, "training", False)
    model.eval()
    writer = writer if writer is not None else getattr(model, "writer", None)

    total_structure_score = 0.0
    total_num_tags = sum(sum(batch.lengths) for batch in batches)
    try:
        with torch.no_grad():
            for batch_index, batch in enumerate(batches):
                structure_score = model.likelihood(
                    batch.word_ids,
                    batch.variable_chars,
                    batch.sentence_indices,
                    set_grammar=(batch_index == 0),
                    lengths=batch.lengths,
                )
                total_structure_score += float(structure_score)
    finally:
        if was_training:
            model.train()

    average_structure_score = (
        total_structure_score / total_num_tags if total_num_tags > 0 else 0.0
    )
    if writer is not None:
        writer.add_scalar(
            f"{section}_epochwise/average_structure_loss",
            average_structure_score,
            epoch,
        )
    if logger is not None:
        logger.info(
            "Epoch %s EVALUATION | Structure loss %.4f",
            epoch,
            total_structure_score,
        )
    return total_structure_score


def write_predicted_trees(
    trees: Sequence[object],
    original_sentences: Sequence[Sentence],
    tree_path: Path | str,
) -> List[object]:
    """Write predicted trees with original tokens restored as leaves."""

    if len(trees) != len(original_sentences):
        raise ValueError(
            "Tree/sentence count mismatch: "
            f"{len(trees)} trees for {len(original_sentences)} sentences."
        )

    tree_path = Path(tree_path)
    tree_path.parent.mkdir(parents=True, exist_ok=True)
    reconstituted_trees = []
    with gzip.open(tree_path, "wt", encoding="utf-8") as ofh:
        for sentence_index, (tree, sentence) in enumerate(zip(trees, original_sentences)):
            if tree is None:
                print(f"{sentence_index}#!#!", file=ofh)
                continue

            output_tree = tree.copy(deep=True)
            content_tokens = _content_tokens(sentence)
            if len(output_tree.leaves()) != len(content_tokens):
                raise ValueError(
                    "\n{}: {}\n{}".format(
                        sentence_index,
                        output_tree,
                        " ".join(content_tokens),
                    )
                )
            for leaf_index, position in enumerate(output_tree.treepositions("leaves")):
                output_tree[position] = content_tokens[leaf_index]
            print(
                f"{sentence_index}#!#!{output_tree.pformat(margin=10000)}",
                file=ofh,
            )
            reconstituted_trees.append(output_tree)
    return reconstituted_trees


def evaluate_trees(
    predicted_trees: Sequence[object],
    gold_tree_strings: Sequence[str],
    writer=None,
    epoch: Optional[int] = None,
    section: str = "dev",
    logger: Optional[logging.Logger] = None,
) -> EvaluationMetrics:
    """Compute unlabeled tree metrics and optionally write TensorBoard scalars."""

    gold_trees = [nltk.Tree.fromstring(tree_string) for tree_string in gold_tree_strings]
    with _route_eval_info_logs_to(logger):
        precision, recall, f1, homogeneity, rh, vm = eval_rvm_et_al(
            (gold_trees, list(predicted_trees))
        )
    metrics = EvaluationMetrics(
        precision=precision,
        recall=recall,
        f1=f1,
        homogeneity=homogeneity,
        rh=rh,
        vm=vm,
    )

    if writer is not None and epoch is not None:
        writer.add_scalar(f"{section}_epochwise/p", precision, epoch)
        writer.add_scalar(f"{section}_epochwise/r", recall, epoch)
        writer.add_scalar(f"{section}_epochwise/f1", f1, epoch)
        writer.add_scalar(f"{section}_epochwise/homogeneity", homogeneity, epoch)
        writer.add_scalar(f"{section}_epochwise/rh", rh, epoch)
        writer.add_scalar(f"{section}_epochwise/vm", vm, epoch)
    return metrics


def evaluate_dataset(
    model,
    batches: Sequence[Batch],
    original_sentences: Sequence[Sentence],
    tree_path: Path | str,
    epoch: int,
    gold_tree_strings: Optional[Sequence[str]] = None,
    writer=None,
    section: str = "dev",
    logger: Optional[logging.Logger] = None,
) -> EvaluationResult:
    """Parse, write trees, and optionally compute gold-tree metrics."""

    parse_result = parse_batches(
        model,
        batches,
        epoch,
        writer=writer,
        section=section,
        logger=logger,
    )
    predicted_trees = write_predicted_trees(
        parse_result.trees,
        original_sentences,
        tree_path,
    )
    metrics = None
    if gold_tree_strings is not None:
        metrics = evaluate_trees(
            predicted_trees,
            gold_tree_strings,
            writer=writer,
            epoch=epoch,
            section=section,
            logger=logger,
        )
    return EvaluationResult(
        parse_result=parse_result,
        tree_path=Path(tree_path),
        predicted_trees=predicted_trees,
        metrics=metrics,
    )


def offline_eval(gold_trees_path: Path | str, predicted_trees_path: Path | str) -> EvaluationMetrics:
    """Evaluate gold/predicted tree files from disk."""

    precision, recall, f1, homogeneity, rh, vm = eval_rvm_et_al(
        ["--gold", str(gold_trees_path), "--pred", str(predicted_trees_path)]
    )
    return EvaluationMetrics(
        precision=precision,
        recall=recall,
        f1=f1,
        homogeneity=homogeneity,
        rh=rh,
        vm=vm,
    )


def _empty_tree_list(batches: Sequence[Batch]) -> List[object]:
    max_index = -1
    for batch in batches:
        if batch.sentence_indices:
            max_index = max(max_index, max(batch.sentence_indices))
    return [None] * (max_index + 1)


def _content_tokens(sentence: Sentence) -> List[str]:
    if len(sentence) >= 2 and sentence[0] == BOS and sentence[-1] == EOS:
        return sentence[1:-1]
    return list(sentence)


@contextmanager
def _route_eval_info_logs_to(logger: Optional[logging.Logger]):
    if logger is None:
        yield
        return

    old_info = logging.info

    def forwarded_info(message, *args, **kwargs):
        del kwargs
        if args:
            message = message % args
        logger.info(message)

    logging.info = forwarded_info
    try:
        yield
    finally:
        logging.info = old_info
