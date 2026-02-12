from parser_utils import apply_cli_overrides
from sounder import Sounder, SounderConfig
from sounder_configs import FR3SpectrumSweepConfig

if __name__ == "__main__":
    config = FR3SpectrumSweepConfig()
    apply_cli_overrides(config, SounderConfig)

    sounder = Sounder(config)
    sounder.run()
