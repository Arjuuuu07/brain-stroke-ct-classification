"""YAML config loading: default.yaml + optional local.yaml override + CLI overrides."""
from pathlib import Path

import yaml


def _deep_merge(base, override):
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _resolve_paths(cfg):
    p = cfg["paths"]
    root = Path(p["data_root"])

    def join(x):
        x = Path(x)
        return str(x if x.is_absolute() else root / x)

    p["folders"] = {k: join(v) for k, v in p["folders"].items()}
    for key in ("external_test_dicom_dir", "external_test_labels_csv"):
        p[key] = join(p[key])
    p["output_dir"] = str(Path(p["output_dir"]))


def load_config(config_path="configs/default.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    local_path = Path(config_path).with_name("local.yaml")
    if local_path.exists():
        with open(local_path, "r", encoding="utf-8") as f:
            cfg = _deep_merge(cfg, yaml.safe_load(f) or {})

    _resolve_paths(cfg)
    return cfg


def apply_overrides(section, args, names):
    """Copy CLI args into a config section. None / False means 'not provided'."""
    for name in names:
        v = getattr(args, name, None)
        if v is not None and v is not False:
            section[name] = v


def run_name(cfg):
    """Name used for every output file of a run, e.g. cv5_t2_resnet18_mytag."""
    tc = cfg["training"]
    tag = f"_{tc['tag']}" if tc.get("tag") else ""
    return f"cv{tc['folds']}_{tc['channels']}_{tc['backbone']}{tag}"


def channel_indices(cfg):
    """Plane indices for the chosen channel preset (training.channels)."""
    name = cfg["training"]["channels"]
    presets = cfg["channels"]
    if name not in presets:
        raise ValueError(f"unknown channels '{name}', choose from {list(presets)}")
    return tuple(presets[name])
