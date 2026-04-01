from parser_utils import apply_cli_overrides
from sigcom_toolkit.base import DebugMode
from sounder import Sounder, SounderConfig
from sounder_configs import FR3RoboticLocalizationConfig, FR3SpectrumSweepConfig, TestConfig

config = TestConfig()
apply_cli_overrides(config, SounderConfig)
# config.debug_mode = DebugMode.HIGH

sounder = Sounder(config)
sounder.run()
