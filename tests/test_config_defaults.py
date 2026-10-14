"""The dataclass defaults ARE the default configuration: Config() == configs/default.yaml (audit C2)."""
import dataclasses

from dlent_rt.config import Config, DEFAULT_CONFIG_PATH, default_config, resolve_path, REPO_ROOT


def _diff(a, b, path=""):
    if dataclasses.is_dataclass(a):
        return [d for f in dataclasses.fields(a)
                for d in _diff(getattr(a, f.name), getattr(b, f.name), f"{path}.{f.name}".lstrip("."))]
    return [] if (a == b and type(a) is type(b)) else [f"{path}: dataclass={a!r} yaml={b!r}"]


def test_dataclass_defaults_equal_yaml():
    assert DEFAULT_CONFIG_PATH.is_file()
    assert _diff(Config().validate(), default_config()) == []
    assert Config().validate() == default_config()


def test_data_paths_relative_to_repo_root():
    cfg = default_config()
    assert not cfg.data.batch_csv.startswith("/")                 # no machine-specific path in the YAML
    assert resolve_path(cfg.data.batch_csv) == REPO_ROOT / cfg.data.batch_csv
    assert str(resolve_path("/abs/path.csv")) == "/abs/path.csv"
