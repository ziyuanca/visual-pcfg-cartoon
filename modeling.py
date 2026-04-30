"""Model and parser construction"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional

import torch
import torch.optim as optim

from pcfgs import CompoundPCFG, SimpleCompPCFGCharNoDistinction
from top_models import TopModel

from config import CompoundParserConfig, PipelineConfig, SimpleParserConfig
from data import Vocabularies


@dataclass(frozen=True)
class ModelBundle:
    """Constructed model objects and useful metadata."""

    parser: torch.nn.Module
    model: TopModel
    parameter_count: int


def build_parser(config: PipelineConfig, vocabularies: Vocabularies) -> torch.nn.Module:
    """Build the configured parser from vocabulary sizes and typed config."""

    num_words = len(vocabularies.word_to_id)
    num_chars = len(vocabularies.char_to_id)

    if isinstance(config.parser, SimpleParserConfig):
        return SimpleCompPCFGCharNoDistinction(
            state_dim=config.state_dim,
            num_states=config.parser.num_states,
            num_chars=num_chars,
            device=config.device,
            eval_device=config.eval_device,
            model_type=config.model_type,
            num_words=num_words,
            rnn_hidden_dim=config.rnn_hidden_dim,
        )

    if isinstance(config.parser, CompoundParserConfig):
        return CompoundPCFG(
            state_dim=config.state_dim,
            nt_states=config.parser.nt_states,
            t_states=config.parser.t_states,
            z_dim=config.parser.z_dim,
            h_dim=config.parser.h_dim,
            w_dim=config.parser.w_dim,
            device=config.device,
            eval_device=config.eval_device,
            model_type=config.model_type,
            num_words=num_words,
            num_chars=num_chars,
            rnn_hidden_dim=config.rnn_hidden_dim,
        )

    raise TypeError(f"Unsupported parser config: {type(config.parser).__name__}")


def build_top_model(
    parser: torch.nn.Module,
    config: PipelineConfig,
    vocabularies: Vocabularies,
    writer=None,
    image_dim: Optional[int] = None,
) -> TopModel:
    """Build the top-level model wrapper."""

    if config.joint_training and image_dim is None:
        raise ValueError("image_dim is required when joint_training=true.")

    return TopModel(
        parser,
        writer,
        config=_TopModelConfigAdapter(config),
        vocab_size=len(vocabularies.word_to_id),
        image_dim=image_dim,
    )


def build_model_bundle(
    config: PipelineConfig,
    vocabularies: Vocabularies,
    writer=None,
    eval_only: bool = False,
    load_checkpoint_state: bool = True,
    validate_devices: bool = True,
    image_dim: Optional[int] = None,
) -> ModelBundle:
    """Build parser and top-level model, optionally loading a checkpoint."""


    parser = build_parser(config, vocabularies)
    model = build_top_model(
        parser,
        config,
        vocabularies,
        writer=writer,
        image_dim=image_dim,
    )
    model = model.to(config.device)

    checkpoint_path = resolve_checkpoint_path(config, eval_only=eval_only)
    if load_checkpoint_state and checkpoint_path is not None:
        load_model_checkpoint(model, checkpoint_path)

    return ModelBundle(
        parser=parser,
        model=model,
        parameter_count=count_parameters(model),
    )


def build_optimizer(config: PipelineConfig, model: torch.nn.Module) -> optim.Optimizer:
    if config.optimizer != "adam":
        raise ValueError(f"Unsupported optimizer: {config.optimizer}")
    return optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        betas=(0.75, 0.999),
    )


def count_parameters(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def resolve_checkpoint_path(
    config: PipelineConfig,
    eval_only: bool = False,
) -> Optional[Path]:
    if eval_only:
        if config.model_path is None:
            raise ValueError("model_path is required to infer eval checkpoint path.")
        return Path(config.model_path) / "model.pth"
    return None


def load_model_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: Path | str,
    map_location: str = "cpu",
) -> None:
    checkpoint_path = Path(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    model.load_state_dict(checkpoint)


def save_model_checkpoint(model: torch.nn.Module, checkpoint_path: Path | str) -> None:
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), checkpoint_path)


class _TopModelConfigAdapter:
    """Small ConfigParser-style adapter for the existing root-level TopModel."""

    def __init__(self, config: PipelineConfig):
        self._values = config.to_flat_dict()

    def getboolean(self, key: str, fallback=None):
        value = self._get(key, fallback)
        if isinstance(value, bool):
            return value
        if value in ("true", "false"):
            return value == "true"
        raise ValueError(f"{key} must be boolean-compatible; found {value!r}.")

    def getfloat(self, key: str, fallback=None):
        return float(self._get(key, fallback))

    def getint(self, key: str, fallback=None):
        return int(self._get(key, fallback))

    def _get(self, key: str, fallback):
        return self._values[key] if key in self._values else fallback
