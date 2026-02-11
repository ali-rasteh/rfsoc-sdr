import argparse
import json


def _parse_value(raw_value, current_value):
    if isinstance(current_value, bool):
        value = raw_value.strip().lower()
        if value in {"1", "true", "yes", "y", "on"}:
            return True
        if value in {"0", "false", "no", "n", "off"}:
            return False
        raise ValueError(f"Invalid boolean value: {raw_value}")
    if isinstance(current_value, int) and not isinstance(current_value, bool):
        return int(raw_value)
    if isinstance(current_value, float):
        return float(raw_value)
    if isinstance(current_value, (list, dict)):
        return json.loads(raw_value)
    return raw_value


def _build_arg_parser(config_class):
    parser = argparse.ArgumentParser(description="RFSoC sounder runner")
    for name in dir(config_class):
        if name.startswith("_"):
            continue
        parser.add_argument(f"--{name}", default=None)
    return parser


def apply_cli_overrides(config, config_class=None):
    if config_class is None:
        config_class = type(config)
    parser = _build_arg_parser(config_class)
    args = parser.parse_args()
    allowed = {name for name in dir(config_class) if not name.startswith("_")}
    for key, raw_value in vars(args).items():
        if raw_value is None:
            continue
        if key in allowed and hasattr(config, key):
            current_value = getattr(config, key)
            setattr(config, key, _parse_value(raw_value, current_value))
