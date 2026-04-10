from parser_utils import apply_cli_overrides
from sounder import Sounder, SounderConfig
from sounder_configs import TestConfig, FR3RoboticLocalizationConfig


def main():
    config = FR3RoboticLocalizationConfig()
    apply_cli_overrides(config, SounderConfig)

    # from sigcom_toolkit.base import DebugMode
    # config.debug_mode = DebugMode.HIGH

    sounder = Sounder(config)
    sounder.run()

if __name__ == "__main__":
    main()
