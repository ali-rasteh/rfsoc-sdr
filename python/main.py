from parser_utils import apply_cli_overrides
from sounder import Sounder, SounderConfig
from sounder_configs import FR3SpectrumSweepConfig

config = FR3SpectrumSweepConfig()
apply_cli_overrides(config, SounderConfig)

print(config.action_loop)
sounder = Sounder(config)
sounder.run()
