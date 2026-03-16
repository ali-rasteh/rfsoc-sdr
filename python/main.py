from parser_utils import apply_cli_overrides
from sounder import Sounder, SounderConfig
from sounder_configs import FR3SpectrumSweepConfig, FR3RoboticLocalizationConfig

# config = FR3SpectrumSweepConfig()
config = FR3RoboticLocalizationConfig()
apply_cli_overrides(config, SounderConfig)

sounder = Sounder(config)
sounder.run()
