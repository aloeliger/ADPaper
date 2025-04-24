import json
import os

CMSSW_BASE_ = os.getenv('CMSSW_BASE')
CONFIG_PATH_ = f'{CMSSW_BASE_}/src//ADPaper/Plots/scripts/nano_paper_scripts/configuration/config.json'

class Configuration():
    def __init__(self):
        with open(CONFIG_PATH_) as the_file:
            self.configs = json.load(the_file)
            
    @staticmethod
    def GetConfiguration():
        return Configuration()
