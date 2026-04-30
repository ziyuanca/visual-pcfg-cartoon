"""Typed configuration"""

from __future__ import annotations

from configparser import ConfigParser
from dataclasses import asdict, dataclass, field
from typing import Dict, Iterable, Mapping, Optional, Set, Tuple, Union

from data import DataPipelineConfig


@dataclass(frozen=True)
class SimpleParserConfig:
    num_states: int = 90

    @property
    def parser_type(self) -> str:
        return "simple"


@dataclass(frozen=True)
class CompoundParserConfig:
    nt_states: int = 30
    t_states: int = 30
    z_dim: int = 64
    h_dim: int = 512
    w_dim: int = 256

    @property
    def parser_type(self) -> str:
        return "compound"


ParserConfig = Union[SimpleParserConfig, CompoundParserConfig]


@dataclass(frozen=True)
class PipelineConfig:
    """Resolved train/eval configuration.

    The parser-specific fields live in ``parser``.  Common fields are kept flat
    because training, data, model construction, and evaluation all consume them.
    """

    parser: ParserConfig = field(default_factory=SimpleParserConfig)
    model: str = "my_model"
    seed: int = -1
    device: str = "cpu"
    eval_device: str = "cpu"
    optimizer: str = "adam"
    max_grad_norm: float = 5.0
    learning_rate: float = 0.0001
    batch_size: int = 2
    max_vocab_size: int = 150000
    eval_steps: int = 2
    eval_start_epoch: int = 1
    start_epoch: int = 0
    max_epoch: int = 20
    logfile: str = "log.txt"
    model_type: str = "word"
    group_by_length: bool = False
    rnn_hidden_dim: int = 512
    state_dim: int = 64
    eval_patient: int = 5
    train_sents: Optional[str] = None
    valid_sents: Optional[str] = None
    valid_trees: Optional[str] = None
    train_image_embs: Optional[str] = None
    train_image_ids: Optional[str] = None
    joint_training: bool = False
    vse_mt_alpha: float = 0.0
    vse_lm_alpha: float = 1.0
    margin: float = 0.2
    word_dim: int = 256
    lstm_dim: int = 256
    sem_dim: int = 256
    syn_dim: int = 256
    no_imgnorm: bool = False
    vectorized_span_matching: bool = False
    span_matching_chunk_size: int = 0
    model_path: Optional[str] = None

    @property
    def parser_type(self) -> str:
        return self.parser.parser_type

    def to_data_config(self) -> DataPipelineConfig:
        if self.train_sents is None:
            raise ValueError("train_sents is required to prepare data.")
        return DataPipelineConfig(
            train_sents=self.train_sents,
            valid_sents=self.valid_sents,
            valid_trees=self.valid_trees,
            train_image_embs=self.train_image_embs,
            train_image_ids=self.train_image_ids,
            batch_size=self.batch_size,
            max_vocab_size=self.max_vocab_size,
            min_count=1,
            device=self.device,
            eval_device=self.eval_device,
            parser_type=self.parser_type,
            group_by_length=self.group_by_length,
            shuffle=True,
            sort=True,
        )

    def to_flat_dict(self) -> Dict[str, object]:
        values = asdict(self)
        parser_values = values.pop("parser")
        values["parser_type"] = self.parser_type
        values.update(parser_values)
        return values

    def write_ini(self, path: str) -> None:
        parser = ConfigParser()
        parser["DEFAULT"] = {
            key: _format_ini_value(value)
            for key, value in self.to_flat_dict().items()
        }
        with open(path, "w", encoding="utf-8") as fh:
            parser.write(fh)


COMMON_DEFAULTS: Dict[str, str] = {
    "model": "my_model",
    "seed": "-1",
    "device": "cpu",
    "eval_device": "cpu",
    "optimizer": "adam",
    "max_grad_norm": "5",
    "learning_rate": "0.0001",
    "batch_size": "2",
    "max_vocab_size": "150000",
    "eval_steps": "2",
    "eval_start_epoch": "1",
    "start_epoch": "0",
    "max_epoch": "20",
    "logfile": "log.txt",
    "model_type": "word",
    "parser_type": "simple",
    "group_by_length": "false",
    "rnn_hidden_dim": "512",
    "state_dim": "64",
    "eval_patient": "5",
    "train_sents": "",
    "valid_sents": "",
    "valid_trees": "",
    "train_image_embs": "",
    "train_image_ids": "",
    "joint_training": "false",
    "vse_mt_alpha": "0.0",
    "vse_lm_alpha": "1.0",
    "margin": "0.2",
    "word_dim": "256",
    "lstm_dim": "256",
    "sem_dim": "256",
    "syn_dim": "256",
    "no_imgnorm": "false",
    "vectorized_span_matching": "false",
    "span_matching_chunk_size": "0",
    "model_path": "",
}

PARSER_DEFAULTS: Dict[str, Dict[str, str]] = {
    "simple": {
        "num_states": "90",
    },
    "compound": {
        "nt_states": "30",
        "t_states": "30",
        "z_dim": "64",
        "h_dim": "512",
        "w_dim": "256",
    },
}

INCOMPATIBLE_PARSER_KEYS: Dict[str, Set[str]] = {
    "simple": {"nt_states", "t_states", "z_dim", "h_dim", "w_dim"},
    "compound": {"num_states"},
}

COMMON_KEYS = set(COMMON_DEFAULTS)
PARSER_KEYS = {key for defaults in PARSER_DEFAULTS.values() for key in defaults}
ALLOWED_KEYS = COMMON_KEYS | PARSER_KEYS


def load_config(
    config_path: Optional[str] = None,
    overrides: Optional[Iterable[str]] = None,
    eval_only: bool = False,
) -> PipelineConfig:
    """Load, merge, type, and validate pipeline configuration."""

    overrides = list(overrides or [])
    file_values, explicit_file_keys = _read_config_file(config_path)
    override_values = parse_overrides(overrides)
    explicit_keys = set(explicit_file_keys) | set(override_values)

    parser_type = _infer_parser_type(file_values, override_values)
    _validate_parser_type(parser_type)
    _validate_known_keys(explicit_keys)
    _validate_parser_keys(parser_type, explicit_keys)

    raw = dict(COMMON_DEFAULTS)
    raw.update(PARSER_DEFAULTS[parser_type])
    raw.update(file_values)
    raw.update(override_values)
    raw["parser_type"] = parser_type

    config = _build_pipeline_config(raw)
    validate_config(config, eval_only=eval_only)
    return config


def parse_overrides(overrides: Iterable[str]) -> Dict[str, str]:
    values = {}
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"Invalid override {override!r}; expected key=value.")
        key, value = override.split("=", 1)
        key = key.strip().lower()
        if not key:
            raise ValueError(f"Invalid override {override!r}; key is empty.")
        values[key] = value.strip()
    return values


def validate_config(config: PipelineConfig, eval_only: bool = False) -> None:
    if config.batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if config.max_vocab_size <= 0:
        raise ValueError("max_vocab_size must be positive.")
    if config.max_epoch < config.start_epoch:
        raise ValueError("max_epoch must be >= start_epoch.")
    if config.eval_steps <= 0:
        raise ValueError("eval_steps must be positive.")
    if config.eval_patient <= 0:
        raise ValueError("eval_patient must be positive.")
    if config.span_matching_chunk_size < 0:
        raise ValueError("span_matching_chunk_size must be non-negative.")

    if config.joint_training:
        if config.vse_mt_alpha <= 0.0:
            raise ValueError("joint_training=true requires vse_mt_alpha > 0.0.")
        if not eval_only and config.train_image_embs is None:
            raise ValueError(
                "joint_training=true requires train_image_embs and train_image_ids."
            )
    else:
        if config.vse_mt_alpha != 0.0:
            raise ValueError("joint_training=false requires vse_mt_alpha=0.0.")
        if config.train_image_embs is not None:
            joined = "train_image_*"
            raise ValueError(
                f"joint_training=false does not accept image inputs: {joined}."
            )


def _read_config_file(config_path: Optional[str]) -> Tuple[Dict[str, str], Set[str]]:
    if config_path is None:
        return {}, set()

    parser = ConfigParser()
    read_paths = parser.read(config_path)
    if not read_paths:
        raise FileNotFoundError(config_path)

    values = {key.lower(): value for key, value in parser.defaults().items()}
    return values, set(values)


def _infer_parser_type(
    file_values: Mapping[str, str],
    override_values: Mapping[str, str],
) -> str:
    parser_type = COMMON_DEFAULTS["parser_type"]
    if "parser_type" in file_values:
        parser_type = file_values["parser_type"]
    if "parser_type" in override_values:
        parser_type = override_values["parser_type"]
    return parser_type.strip().lower()


def _validate_parser_type(parser_type: str) -> None:
    if parser_type not in PARSER_DEFAULTS:
        raise ValueError("Unknown parser_type: {}. Use 'simple' or 'compound'.".format(parser_type))


def _validate_known_keys(explicit_keys: Set[str]) -> None:
    unknown = explicit_keys - ALLOWED_KEYS
    if unknown:
        raise ValueError("Unknown config keys: {}.".format(", ".join(sorted(unknown))))


def _validate_parser_keys(parser_type: str, explicit_keys: Set[str]) -> None:
    incompatible = explicit_keys.intersection(INCOMPATIBLE_PARSER_KEYS[parser_type])
    if incompatible:
        raise ValueError(
            "parser_type={} does not accept config keys: {}.".format(
                parser_type, ", ".join(sorted(incompatible))
            )
        )


def _build_pipeline_config(raw: Mapping[str, str]) -> PipelineConfig:
    parser_type = _as_choice(raw["parser_type"], {"simple", "compound"}, "parser_type")
    if parser_type == "simple":
        parser_config: ParserConfig = SimpleParserConfig(
            num_states=_as_int(raw["num_states"], "num_states")
        )
    else:
        parser_config = CompoundParserConfig(
            nt_states=_as_int(raw["nt_states"], "nt_states"),
            t_states=_as_int(raw["t_states"], "t_states"),
            z_dim=_as_int(raw["z_dim"], "z_dim"),
            h_dim=_as_int(raw["h_dim"], "h_dim"),
            w_dim=_as_int(raw["w_dim"], "w_dim"),
        )

    return PipelineConfig(
        parser=parser_config,
        model=_as_nonempty_str(raw["model"], "model"),
        seed=_as_int(raw["seed"], "seed"),
        device=_as_choice(raw["device"], {"cpu", "cuda"}, "device"),
        eval_device=_as_choice(raw["eval_device"], {"cpu", "cuda"}, "eval_device"),
        optimizer=_as_choice(raw["optimizer"], {"adam"}, "optimizer"),
        max_grad_norm=_as_float(raw["max_grad_norm"], "max_grad_norm"),
        learning_rate=_as_float(raw["learning_rate"], "learning_rate"),
        batch_size=_as_int(raw["batch_size"], "batch_size"),
        max_vocab_size=_as_int(raw["max_vocab_size"], "max_vocab_size"),
        eval_steps=_as_int(raw["eval_steps"], "eval_steps"),
        eval_start_epoch=_as_int(raw["eval_start_epoch"], "eval_start_epoch"),
        start_epoch=_as_int(raw["start_epoch"], "start_epoch"),
        max_epoch=_as_int(raw["max_epoch"], "max_epoch"),
        logfile=_as_nonempty_str(raw["logfile"], "logfile"),
        model_type=_as_choice(raw["model_type"], {"word", "char"}, "model_type"),
        group_by_length=_as_bool(raw["group_by_length"], "group_by_length"),
        rnn_hidden_dim=_as_int(raw["rnn_hidden_dim"], "rnn_hidden_dim"),
        state_dim=_as_int(raw["state_dim"], "state_dim"),
        eval_patient=_as_int(raw["eval_patient"], "eval_patient"),
        train_sents=_as_optional_str(raw["train_sents"]),
        valid_sents=_as_optional_str(raw["valid_sents"]),
        valid_trees=_as_optional_str(raw["valid_trees"]),
        train_image_embs=_as_optional_str(raw["train_image_embs"]),
        train_image_ids=_as_optional_str(raw["train_image_ids"]),
        joint_training=_as_bool(raw["joint_training"], "joint_training"),
        vse_mt_alpha=_as_float(raw["vse_mt_alpha"], "vse_mt_alpha"),
        vse_lm_alpha=_as_float(raw["vse_lm_alpha"], "vse_lm_alpha"),
        margin=_as_float(raw["margin"], "margin"),
        word_dim=_as_int(raw["word_dim"], "word_dim"),
        lstm_dim=_as_int(raw["lstm_dim"], "lstm_dim"),
        sem_dim=_as_int(raw["sem_dim"], "sem_dim"),
        syn_dim=_as_int(raw["syn_dim"], "syn_dim"),
        no_imgnorm=_as_bool(raw["no_imgnorm"], "no_imgnorm"),
        vectorized_span_matching=_as_bool(
            raw["vectorized_span_matching"], "vectorized_span_matching"
        ),
        span_matching_chunk_size=_as_int(
            raw["span_matching_chunk_size"], "span_matching_chunk_size"
        ),
        model_path=_as_optional_str(raw["model_path"]),
    )


def _as_optional_str(value: str) -> Optional[str]:
    value = value.strip()
    return value or None


def _as_nonempty_str(value: str, key: str) -> str:
    value = value.strip()
    if not value:
        raise ValueError(f"{key} must not be empty.")
    return value


def _as_int(value: str, key: str) -> int:
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"{key} must be an integer; found {value!r}.") from exc


def _as_float(value: str, key: str) -> float:
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"{key} must be a float; found {value!r}.") from exc


def _as_bool(value: str, key: str) -> bool:
    normalized = value.strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise ValueError(f"{key} must be either 'true' or 'false'; found {value!r}.")


def _as_choice(value: str, choices: Set[str], key: str) -> str:
    normalized = value.strip().lower()
    if normalized not in choices:
        raise ValueError(
            "{} must be one of {}; found {!r}.".format(
                key, ", ".join(sorted(choices)), value
            )
        )
    return normalized


def _format_ini_value(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)
