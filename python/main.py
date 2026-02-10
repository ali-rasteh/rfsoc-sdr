from sounder import Sounder
from sounder_configs import *



if __name__ == '__main__':
    
    config = FR3SpectrumSweepConfig()
    sounder = Sounder(config)
    sounder.run()


