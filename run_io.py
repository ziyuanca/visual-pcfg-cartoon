"""Run-directory, logging, and artifact I/O"""

from __future__ import annotations

import gzip
import logging
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, TextIO

from config import PipelineConfig
from data import Vocabularies, write_vocabularies


@dataclass(frozen=True)
class RunPaths:
    """Canonical artifact paths for one run."""

    run_dir: Path
    config_ini: Path
    log_file: Path
    tensorboard_dir: Path
    word_dictionary: Path
    char_dictionary: Path
    model_checkpoint: Path

    def tree_file(self, epoch: int) -> Path:
        return self.run_dir / f"e{epoch}.vittrees.gz"


@dataclass
class RunIO:
    """Open IO resources for a run."""

    paths: RunPaths
    logger: logging.Logger
    log_handle: TextIO
    writer: Optional[object] = None

    def close(self) -> None:
        if self.writer is not None:
            self.writer.close()
            self.writer = None

        for handler in list(self.logger.handlers):
            handler.flush()
            handler.close()
            self.logger.removeHandler(handler)

        self.log_handle.close()

    def __enter__(self) -> "RunIO":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def create_run_paths(
    config: PipelineConfig,
    output_root: os.PathLike | str = "outputs",
    eval_only: bool = False,
    overwrite: bool = False,
) -> RunPaths:
    """Create or validate a run directory and return canonical paths."""

    run_dir = _resolve_run_dir(config, Path(output_root), eval_only=eval_only)
    if eval_only:
        if not run_dir.exists():
            raise FileNotFoundError(f"Run directory does not exist: {run_dir}")
        if not run_dir.is_dir():
            raise NotADirectoryError(str(run_dir))
    else:
        _create_train_run_dir(run_dir, overwrite=overwrite)

    return RunPaths(
        run_dir=run_dir,
        config_ini=run_dir / "config.ini",
        log_file=run_dir / config.logfile,
        tensorboard_dir=run_dir / "tensorboard",
        word_dictionary=run_dir / "word.dic",
        char_dictionary=run_dir / "char.dic",
        model_checkpoint=run_dir / "model.pth",
    )


def setup_run_io(
    config: PipelineConfig,
    output_root: os.PathLike | str = "outputs",
    eval_only: bool = False,
    overwrite: bool = False,
    create_writer: bool = True,
    stdout: bool = True,
) -> RunIO:
    """Create run paths, save config, configure logging, and open writer."""

    paths = create_run_paths(
        config,
        output_root=output_root,
        eval_only=eval_only,
        overwrite=overwrite,
    )
    paths.tensorboard_dir.mkdir(parents=True, exist_ok=True)
    config.write_ini(str(paths.config_ini))

    logger, log_handle = configure_logging(paths.log_file, stdout=stdout)
    writer = create_tensorboard_writer(paths.tensorboard_dir) if create_writer else None

    logger.info("Run directory: %s", paths.run_dir)
    logger.info("Resolved config written to: %s", paths.config_ini)
    return RunIO(paths=paths, logger=logger, log_handle=log_handle, writer=writer)


def configure_logging(
    log_file: os.PathLike | str,
    stdout: bool = True,
    level: int = logging.INFO,
) -> tuple[logging.Logger, TextIO]:
    logger = logging.getLogger("visual_pcfg")
    logger.setLevel(level)
    logger.propagate = False

    for handler in list(logger.handlers):
        handler.flush()
        handler.close()
        logger.removeHandler(handler)

    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_handle = _open_log_file(log_path)

    formatter = logging.Formatter(
        fmt="%(asctime)s %(message)s",
        datefmt="%m/%d/%Y %I:%M:%S %p",
    )
    file_handler = logging.StreamHandler(log_handle)
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    if stdout:
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setLevel(level)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    return logger, log_handle


def _open_log_file(log_path: Path) -> TextIO:
    if log_path.suffix == ".gz":
        return gzip.open(log_path, "wt", encoding="utf-8")
    return open(log_path, "w", encoding="utf-8")


def create_tensorboard_writer(tensorboard_dir: os.PathLike | str):
    """Create a TensorBoard writer lazily to keep import side effects local."""

    from torch.utils.tensorboard import SummaryWriter

    return SummaryWriter(os.fspath(tensorboard_dir), flush_secs=10)


def save_run_config(config: PipelineConfig, paths: RunPaths) -> None:
    paths.run_dir.mkdir(parents=True, exist_ok=True)
    config.write_ini(str(paths.config_ini))


def save_vocabularies(vocabularies: Vocabularies, paths: RunPaths) -> None:
    write_vocabularies(vocabularies, paths.run_dir)


def _resolve_run_dir(
    config: PipelineConfig,
    output_root: Path,
    eval_only: bool,
) -> Path:
    if config.model_path is not None:
        return Path(config.model_path)
    if eval_only:
        raise ValueError("model_path is required for eval/test runs.")

    output_root.mkdir(parents=True, exist_ok=True)
    base = output_root / config.model
    for run_index in range(100):
        candidate = Path(f"{base}_{run_index}")
        if not candidate.exists():
            return candidate
    raise FileExistsError(f"No free run directory found for model {config.model!r}.")


def _create_train_run_dir(run_dir: Path, overwrite: bool) -> None:
    if run_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"Run directory already exists: {run_dir}. "
                "Pass overwrite=True to replace it."
            )
        if not run_dir.is_dir():
            raise NotADirectoryError(str(run_dir))
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=False)
