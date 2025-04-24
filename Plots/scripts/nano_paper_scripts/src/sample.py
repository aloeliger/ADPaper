import os
import ROOT
from .config import Configuration

class NanoSample():
    def __init__(
            self,
            list_of_paths: list[str],
            limit_files: int = None
    ):
        self.list_of_paths = list_of_paths
        self.limit_files = limit_files
        chain, df, nFiles = self.build_sample_()
        self.df = df
        self.chain = chain
        self.nFiles = nFiles

    def build_sample_(self):
        chain = ROOT.TChain('Events')
        nFiles = 0
        for path in self.list_of_paths:
            for root, dirs, files in os.walk(path):
                for fileName in files:
                    theFile = f'{root}/{fileName}'
                    if self.limit_files is not None and nFiles >= self.limit_files:
                        break
                    nFiles += 1
                    chain.Add(theFile)
                if self.limit_files is not None and nFiles >= self.limit_files:
                    break
        dataframe = ROOT.RDataFrame(chain)
        return chain, dataframe, nFiles

def construct_data_samples(limit_files=None):
    data_paths = Configuration.GetConfiguration().configs['data_paths']

    sample_collection = {}
    for sample in data_paths:
        sample_collection[sample] = NanoSample(
            data_paths[sample],
            limit_files=limit_files
        )
    return sample_collection

def construct_mc_samples(limit_files=None):
    mc_paths = Configuration.GetConfiguration().configs['mc_paths']

    sample_collection = {}
    for sample in mc_paths:
        sample_collection[sample] = NanoSample(
            mc_paths[sample],
            limit_files=limit_files
        )
    return sample_collection
