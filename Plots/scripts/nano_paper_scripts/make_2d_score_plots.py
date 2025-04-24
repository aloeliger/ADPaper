import src.score_plots_2D as score_plots_2D

import mplhep as hep
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
import ROOT
import itertools
from pathlib import Path
from rich.console import Console

from src.sample import construct_data_samples, construct_mc_samples
from src.config import Configuration

console = Console()

def main():
    console.log("Making 2D scatter plots")

    current_config = Configuration.GetConfiguration().configs

    output_path = Path(current_config['output path']+'/2D_score_plots/')
    output_path.mkdir(exist_ok=True, parents=True)
    
    data_sample = construct_data_samples()['RunI']
    mc_samples = construct_mc_samples()
    
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
    main()
