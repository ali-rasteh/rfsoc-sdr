from parser_utils import apply_cli_overrides
from sigcom_toolkit.general import DebugMode
from sounder import Sounder, SounderConfig
from sounder_configs import FR3RoboticLocalizationConfig, FR3SpectrumSweepConfig

config = FR3SpectrumSweepConfig()
# config = FR3RoboticLocalizationConfig()
apply_cli_overrides(config, SounderConfig)
# config.debug_mode = DebugMode.HIGH

sounder = Sounder(config)
sounder.run()
