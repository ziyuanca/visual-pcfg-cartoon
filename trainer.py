"""Training loop"""

from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np
import torch

from config import PipelineConfig
from data import Batch, Sentence
from evaluator import evaluate_dataset, likelihood_batches
from modeling import build_optimizer, save_model_checkpoint


@dataclass(frozen=True)
class TrainEpochResult:
    epoch: int
    total_loss: float
    total_tokens: int
    average_loss: float
    batch_count: int


@dataclass(frozen=True)
class ScheduledEvalResult:
    epoch: int
    score: float
    improved: bool


@dataclass(frozen=True)
class TrainResult:
    best_eval_score: Optional[float]
    best_epoch: Optional[int]
    epochs_completed: int
    stopped_early: bool
    patient: int
    history: List[TrainEpochResult]
    evaluations: List[ScheduledEvalResult]


def seed_everything(seed: int, use_cuda: bool = False) -> int:
    """Seed Python, NumPy, and PyTorch, returning the concrete seed used."""

    if seed < 0:
        seed = int(int(time.time()) * random.random())

    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if use_cuda and torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    return seed


def train_epoch(
    model,
    optimizer: torch.optim.Optimizer,
    batches: Sequence[Batch],
    epoch: int,
    max_grad_norm: float,
    writer=None,
    logger: Optional[logging.Logger] = None,
) -> TrainEpochResult:
    """Train for one epoch over named batches without mutating batch order."""

    if not batches:
        raise ValueError("Cannot train on an empty batch sequence.")

    model.train()
    writer = writer if writer is not None else getattr(model, "writer", None)

    batch_order = list(range(len(batches)))
    random.shuffle(batch_order)
    progress_points = _progress_points(len(batches))

    total_loss = 0.0
    total_tokens = 0
    start_time = time.time()

    for processed_count, batch_index in enumerate(batch_order, start=1):
        batch = batches[batch_index]
        optimizer.zero_grad()

        loss = model.forward(
            batch.word_ids,
            batch.variable_chars,
            images=batch.images,
            lengths=batch.lengths,
        )
        loss.backward()
        total_loss += float(loss.item())
        total_tokens += sum(batch.lengths)

        if max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        average_loss = total_loss / total_tokens if total_tokens > 0 else 0.0
        global_step = len(batches) * epoch + processed_count
        if processed_count in progress_points:
            if logger is not None:
                logger.info(
                    "Epoch=%s iter=%s lr=%.5f train loss=%.4f time=%.2fs",
                    epoch,
                    processed_count,
                    optimizer.param_groups[0]["lr"],
                    average_loss,
                    time.time() - start_time,
                )
            start_time = time.time()
            if writer is not None:
                writer.add_scalar(
                    "train_accumulative/average_total_loss",
                    average_loss,
                    global_step,
                )

    average_loss = total_loss / total_tokens if total_tokens > 0 else 0.0
    if writer is not None:
        writer.add_scalar("train_epochwise/average_total_loss", average_loss, epoch)

    return TrainEpochResult(
        epoch=epoch,
        total_loss=total_loss,
        total_tokens=total_tokens,
        average_loss=average_loss,
        batch_count=len(batches),
    )


def should_evaluate(epoch: int, config: PipelineConfig) -> bool:
    """Return whether the epoch is scheduled for evaluation."""

    if epoch < config.eval_start_epoch:
        return False
    return (
        (epoch - config.eval_start_epoch) % config.eval_steps == 0
        or epoch + 1 == config.max_epoch
    )


def evaluate_for_training(
    config: PipelineConfig,
    model,
    train_batches: Sequence[Batch],
    epoch: int,
    valid_batches: Optional[Sequence[Batch]] = None,
    valid_sentences: Optional[Sequence[Sentence]] = None,
    valid_trees: Optional[Sequence[str]] = None,
    paths=None,
    writer=None,
    logger: Optional[logging.Logger] = None,
) -> float:
    """Run the scheduled validation objective and return a higher-is-better score."""

    writer = writer if writer is not None else getattr(model, "writer", None)

    if valid_batches is None:
        return -likelihood_batches(
            model,
            train_batches,
            epoch,
            writer=writer,
            logger=logger,
        )

    if valid_sentences is None:
        raise ValueError("valid_sentences is required when valid_batches is provided.")
    if paths is None:
        raise ValueError("paths is required to write validation trees.")

    model.to(config.eval_device)
    try:
        result = evaluate_dataset(
            model,
            valid_batches,
            valid_sentences,
            paths.tree_file(epoch),
            epoch,
            gold_tree_strings=valid_trees,
            writer=writer,
            logger=logger,
        )
        return result.parse_result.total_structure_score
    finally:
        model.to(config.device)


def train(
    config: PipelineConfig,
    model,
    train_batches: Sequence[Batch],
    valid_batches: Optional[Sequence[Batch]] = None,
    valid_sentences: Optional[Sequence[Sentence]] = None,
    valid_trees: Optional[Sequence[str]] = None,
    paths=None,
    writer=None,
    logger: Optional[logging.Logger] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> TrainResult:
    """Run training, scheduled evaluation, best-checkpoint saving, and patience."""

    if optimizer is None:
        optimizer = build_optimizer(config, model)
    writer = writer if writer is not None else getattr(model, "writer", None)

    history: List[TrainEpochResult] = []
    evaluations: List[ScheduledEvalResult] = []
    best_eval_score: Optional[float] = None
    best_epoch: Optional[int] = None
    patient = 0
    stopped_early = False

    for epoch in range(config.start_epoch, config.max_epoch):
        epoch_result = train_epoch(
            model,
            optimizer,
            train_batches,
            epoch,
            config.max_grad_norm,
            writer=writer,
            logger=logger,
        )
        history.append(epoch_result)

        if not should_evaluate(epoch, config):
            continue

        if logger is not None:
            logger.info("EVALING.")
        eval_score = evaluate_for_training(
            config,
            model,
            train_batches,
            epoch,
            valid_batches=valid_batches,
            valid_sentences=valid_sentences,
            valid_trees=valid_trees,
            paths=paths,
            writer=writer,
            logger=logger,
        )

        improved = best_eval_score is None or eval_score > best_eval_score
        evaluations.append(
            ScheduledEvalResult(epoch=epoch, score=eval_score, improved=improved)
        )

        if improved:
            if logger is not None:
                logger.info(
                    "Better model found based on likelihood: %s! vs %s",
                    eval_score,
                    -1e8 if best_eval_score is None else best_eval_score,
                )
            best_eval_score = eval_score
            best_epoch = epoch
            patient = 0
            if paths is not None:
                save_model_checkpoint(model, paths.model_checkpoint)
        else:
            patient += 1
            if patient >= config.eval_patient:
                stopped_early = True
                break

    return TrainResult(
        best_eval_score=best_eval_score,
        best_epoch=best_epoch,
        epochs_completed=len(history),
        stopped_early=stopped_early,
        patient=patient,
        history=history,
        evaluations=evaluations,
    )


def _progress_points(batch_count: int) -> set[int]:
    step = int(batch_count / 10)
    if step <= 0:
        return set()
    return {step * i for i in range(1, 10)}
