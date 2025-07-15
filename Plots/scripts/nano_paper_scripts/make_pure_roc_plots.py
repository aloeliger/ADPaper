import src.pure_roc_plots as pure_roc_plots

import mplhep as hep
import ROOT
import itertools
import numpy as np
from rich.console import Console
import argparse
from pathlib import Path

from src.sample import construct_data_samples, construct_mc_samples
from src.config import Configuration
import src.definitions as definitions

console = Console()

def main(args):
    console.log("Making pure ROCs")

    current_config = Configuration.GetConfiguration().configs

    output_path = Path(current_config['output path']+'/Pure_ROCs/')
    output_path.mkdir(exist_ok=True, parents=True)

    data_samples = construct_data_samples()
    data_sample = data_samples['RunI']
    data_sample.df = data_sample.df.Filter('PV_npvs > 10')

    mc_samples = construct_mc_samples()
    #mc_sample_names = ["GluGluHToGG", "TT", "VBFHTo2b", "HTo2LongLivedTo4b","SingleNeutrino", "VBFHToTauTau", "GluGluHToTauTau"]
    mc_sample_names = ["GluGluHToGG", "TT", "VBFHTo2B", "HTo2LongLivedTo4b","SingleNeutrino",]

    definitions.add_all_values(data_sample)
    definitions.make_collisions_runs_cuts(data_sample)
    for sample in mc_samples:
        definitions.add_all_values(mc_samples[sample])
    
    if args.debug:
        data_sample.df = data_sample.df.Range(1000)
        for sample in mc_samples:
            mc_samples[sample].df = mc_samples[sample].df.Range(1000)

    cicada_names = current_config['CICADA Scores']
    axo_names = current_config['AXO Scores']
    for cicada_name in cicada_names:
        pure_roc_plots.make_pure_roc_plot(
            mc_samples,
            mc_sample_names,
            data_sample,
            'RunI',
            cicada_name,
            cicada_names[cicada_name],
            output_path
        )
    for axo_name in axo_names:
        pure_roc_plots.make_pure_roc_plot(
            mc_samples,
            mc_sample_names,
            data_sample,
            'RunI',
            axo_name,
            axo_names[axo_name],
            output_path
        )
    console.log(
        '[green]Done![/green]'
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Make pure ROC plots")
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Operate on a smaller set, to make checking changes easer'
    )

    args = parser.parse_args()

    main(args)
