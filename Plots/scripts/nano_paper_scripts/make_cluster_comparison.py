import src.cluster_comparison as cluster_comparison

import ROOT
import os
import numpy as np
from rich.console import Console
import argparse
from pathlib import Path

from src.sample import construct_data_samples, construct_mc_samples
from src.config import Configuration
import src.definitions as definitions

console = Console()

def main(args):
    console.log('Making cluster comparison ROCs')

    current_config = Configuration.GetConfiguration().configs

    output_path = Path(current_config['output path']+'/cluster_comparison/')
    output_path.mkdir(exist_ok=True, parents = True)

    data_sample = construct_data_samples()['RunI']
    data_sample.df = data_sample.df.Filter('PV_npvs > 10')

    mc_samples = construct_mc_samples()
    
    single_neutrino_sample = mc_samples.pop('SingleNeutrino', None)

    mc_samples.pop("ZprimeToTauTau", None)
    mc_samples.pop("ZZ", None)
    mc_samples.pop("VBFHToTauTau", None)
    mc_samples.pop("GluGluHToTauTau", None)

    definitions.add_all_values(data_sample)
    definitions.make_collisions_runs_cuts(data_sample)
    for sample in mc_samples:
        definitions.add_all_values(mc_samples[sample])

    if args.debug:
        data_sample.df = data_sample.df.Range(1000)
        single_neutrino_sample.df = single_neutrino_sample.df.Range(1000)
        for sample in mc_samples:
            mc_samples[sample].df = mc_samples[sample].df.Range(1000)

    cicada_names = current_config['CICADA Scores']
    axo_names = current_config['AXO Scores']
    for cicada_name in cicada_names:
        cluster_comparison.make_cluster_comparison_plot(
            background_sample=data_sample,
            background_sample_name='Zero Bias',
            samples=mc_samples,
            score_name=cicada_names[cicada_name],
            score_display_name=cicada_name,
            output_path=output_path,
        )

    for axo_name in axo_names:
        cluster_comparison.make_cluster_comparison_plot(
            background_sample=data_sample,
            background_sample_name='Zero Bias',
            samples=mc_samples,
            score_name=axo_names[axo_name],
            score_display_name=axo_name,
            output_path=output_path,
        )
    console.log('Done with Calo Clusters ROCs')
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Make Cluster comparison ROCs')
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Operate on a smaller set to make checking changes easier'
    )

    args = parser.parse_args()

    main(args)
