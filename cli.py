"""Command-line entrypoint"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
from pprint import pformat
from typing import Optional, Sequence

from config import PipelineConfig, load_config
from data import prepare_data
from evaluator import EvaluationMetrics, evaluate_dataset, offline_eval
from run_io import (
    RunIO,
    configure_logging,
    create_run_paths,
    create_tensorboard_writer,
    save_vocabularies,
    setup_run_io,
)
from modeling import build_model_bundle, build_optimizer
from trainer import TrainResult, seed_everything, train


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="train a model")
    _add_config_args(train_parser)
    train_parser.add_argument("--output-root", default="outputs")
    train_parser.add_argument("--overwrite", action="store_true")
    train_parser.add_argument("--no-tensorboard", action="store_true")
    train_parser.add_argument("--quiet", action="store_true")
    train_parser.set_defaults(func=run_train)

    test_parser = subparsers.add_parser("test", help="evaluate an existing model")
    _add_config_args(test_parser)
    test_parser.add_argument("--model-path")
    test_parser.add_argument("--output-root", default="outputs")
    test_parser.add_argument("--epoch", type=int, default=0)
    test_parser.add_argument("--no-tensorboard", action="store_true")
    test_parser.add_argument("--quiet", action="store_true")
    test_parser.set_defaults(func=run_test)

    offline_parser = subparsers.add_parser(
        "offline-eval",
        help="evaluate existing gold and predicted tree files",
    )
    offline_parser.add_argument("--gold", required=True)
    offline_parser.add_argument("--pred", required=True)
    offline_parser.set_defaults(func=run_offline_eval)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    args.func(args)
    return 0


def run_train(args) -> TrainResult:
    config = load_config(args.config, args.overrides, eval_only=False)
    config = _seed_config(config)

    run_io = setup_run_io(
        config,
        output_root=args.output_root,
        overwrite=args.overwrite,
        create_writer=not args.no_tensorboard,
        stdout=not args.quiet,
    )
    try:
        _log_resolved_config(run_io.logger, config)
        _write_config_text(run_io.writer, config)
        prepared = prepare_data(config.to_data_config())
        save_vocabularies(prepared.vocabularies, run_io.paths)

        bundle = build_model_bundle(
            config,
            prepared.vocabularies,
            writer=run_io.writer,
            eval_only=False,
            image_dim=_train_image_dim(prepared),
        )
        optimizer = build_optimizer(config, bundle.model)
        _log_model_architecture(run_io.logger, bundle.model)
        run_io.logger.info("Parser has %s parameters", bundle.parameter_count)

        return train(
            config,
            bundle.model,
            prepared.train_batches,
            valid_batches=prepared.valid_batches,
            valid_sentences=prepared.valid_sentences,
            valid_trees=prepared.valid_trees,
            paths=run_io.paths,
            writer=run_io.writer,
            logger=run_io.logger,
            optimizer=optimizer,
        )
    finally:
        run_io.close()


def run_test(args):
    config_path = _resolve_test_config_path(args.config, args.model_path)
    overrides = list(args.overrides)
    if args.model_path is not None:
        overrides.append(f"model_path={args.model_path}")

    config = load_config(config_path, overrides, eval_only=True)
    config = _seed_config(config)

    run_io = _open_eval_run_io(
        config,
        output_root=args.output_root,
        create_writer=not args.no_tensorboard,
        stdout=not args.quiet,
    )
    try:
        _log_resolved_config(run_io.logger, config)
        prepared = prepare_data(config.to_data_config())
        if prepared.valid_batches is None or prepared.valid_sentences is None:
            raise ValueError("test requires valid_sents so validation batches can be built.")

        bundle = build_model_bundle(
            config,
            prepared.vocabularies,
            writer=run_io.writer,
            eval_only=True,
            image_dim=_train_image_dim(prepared),
        )
        _log_model_architecture(run_io.logger, bundle.model)
        run_io.logger.info("EVALING.")
        bundle.model.to(config.eval_device)
        try:
            return evaluate_dataset(
                bundle.model,
                prepared.valid_batches,
                prepared.valid_sentences,
                run_io.paths.tree_file(args.epoch),
                args.epoch,
                gold_tree_strings=prepared.valid_trees,
                writer=run_io.writer,
                logger=run_io.logger,
            )
        finally:
            bundle.model.to(config.device)
    finally:
        run_io.close()


def run_offline_eval(args) -> EvaluationMetrics:
    metrics = offline_eval(args.gold, args.pred)
    print(_format_metrics(metrics))
    return metrics


def _add_config_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config")
    parser.add_argument("overrides", nargs="*")


def _seed_config(config: PipelineConfig) -> PipelineConfig:
    seed = seed_everything(config.seed, use_cuda=config.device == "cuda")
    if seed == config.seed:
        return config
    return replace(config, seed=seed)


def _resolve_test_config_path(
    config_path: Optional[str],
    model_path: Optional[str],
) -> Optional[str]:
    if config_path is not None:
        return config_path
    if model_path is None:
        return None
    candidate = Path(model_path) / "config.ini"
    return str(candidate) if candidate.exists() else None


def _open_eval_run_io(
    config: PipelineConfig,
    output_root: str,
    create_writer: bool,
    stdout: bool,
) -> RunIO:
    paths = create_run_paths(config, output_root=output_root, eval_only=True)
    paths.tensorboard_dir.mkdir(parents=True, exist_ok=True)
    logger, log_handle = configure_logging(paths.log_file, stdout=stdout)
    writer = create_tensorboard_writer(paths.tensorboard_dir) if create_writer else None
    logger.info("Run directory: %s", paths.run_dir)
    return RunIO(paths=paths, logger=logger, log_handle=log_handle, writer=writer)


def _write_config_text(writer, config: PipelineConfig) -> None:
    if writer is not None and hasattr(writer, "add_text"):
        writer.add_text("args", str(config.to_flat_dict()))


def _log_resolved_config(logger, config: PipelineConfig) -> None:
    logger.info("Resolved config:\n%s", pformat(config.to_flat_dict(), sort_dicts=True))


def _log_model_architecture(logger, model) -> None:
    logger.info("Model architecture:\n%s", model)


def _train_image_dim(prepared) -> Optional[int]:
    train_images = getattr(prepared, "train_images", None)
    if train_images is None:
        return None
    return int(train_images.embeddings.shape[1])


def _format_metrics(metrics: EvaluationMetrics) -> str:
    return (
        "precision={:.6f} recall={:.6f} f1={:.6f} "
        "homogeneity={:.6f} rh={:.6f} vm={:.6f}"
    ).format(
        metrics.precision,
        metrics.recall,
        metrics.f1,
        metrics.homogeneity,
        metrics.rh,
        metrics.vm,
    )


if __name__ == "__main__":
    raise SystemExit(main())
