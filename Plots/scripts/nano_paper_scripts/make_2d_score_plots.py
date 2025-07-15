import src.score_plots_2D as score_plots_2D

import mplhep as hep
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
import ROOT
import itertools
from pathlib import Path
from rich.console import Console
import argparse

from src.sample import construct_data_samples, construct_mc_samples
from src.config import Configuration
import src.definitions as definitions

console = Console()

def main(args):
    console.log("Making 2D scatter plots")

    current_config = Configuration.GetConfiguration().configs

    output_path = Path(current_config['output path']+'/2D_score_plots/')
    output_path.mkdir(exist_ok=True, parents=True)
    
    data_sample = construct_data_samples()['RunI']
    definitions.make_collisions_runs_cuts(data_sample)
    mc_samples = construct_mc_samples()

    if args.debug:
        data_sample.df = data_sample.df.Range(1000)
        for sample in mc_samples:
            mc_samples[sample].df = mc_samples[sample].df.Range(1000)
    
    cicada_names = current_config['CICADA Scores']
    axo_names = current_config['AXO Scores']
    name_pairs = itertools.product(cicada_names, axo_names)

    for cicada_name, axo_name in name_pairs:
        score_plots_2D.make_scatter_plot(
            data_sample,
            "RunI",
            cicada_name,
            axo_name,
            current_config['CICADA Scores'][cicada_name],
            current_config['AXO Scores'][axo_name],
            output_path
        )
        for mc_sample in mc_samples:
            score_plots_2D.make_scatter_plot(
                mc_samples[mc_sample],
                mc_sample,
                cicada_name,
                axo_name,
                current_config['CICADA Scores'][cicada_name],
                current_config['AXO Scores'][axo_name],
                output_path
            )

    console.log(
        '[green]Done![/green]'
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Make the 2D correlation plots')
    parser.add_argument(
        '--debug',
        action='store_true',
        help='operate on a smaller set, to make checking changes easier'
    )

    args = parser.parse_args()
    
    main(args)
