import mplhep as hep
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
import ROOT

import src.leading_object_scatters as leading_object_scatters
from src.sample import construct_data_samples, construct_mc_samples
from src.config import Configuration
import src.definitions as definitions

from rich.console import Console
from pathlib import Path
import argparse

console = Console()

def main(args):
    console.log('Making leading object correlation plots')

    current_config = Configuration.GetConfiguration().configs

    output_path = Path(current_config['output path']+'/leading_object_scatters/')
    output_path.mkdir(exist_ok=True, parents=True)

    data_sample = construct_data_samples()['RunI']

    if args.debug:
        data_sample.df = data_sample.df.Range(1000)
    definitions.make_collisions_runs_cuts(data_sample)
    definitions.add_all_values(data_sample)

    cicada_names = current_config['CICADA Scores']
    axo_names = current_config['AXO Scores']

    objects = ['Jet', 'Electron', 'Photon', 'Tau']

    for object_name in objects:
        for cicada_name in cicada_names:
            leading_object_scatters.make_leading_object_scatter(
                sample=data_sample,
                sample_name = 'RunI',
                score_name = cicada_names[cicada_name],
                score_display_name = cicada_name,
                object_to_use = object_name,
                output_path = output_path
            )
        for axo_name in axo_names:
            leading_object_scatters.make_leading_object_scatter(
                sample=data_sample,
                sample_name = 'RunI',
                score_name = axo_names[axo_name],
                score_display_name = axo_name,
                object_to_use = object_name,
                output_path = output_path
            )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Make correlation plots to the leading objects')
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Operate on a smaller set to make changes easier'
    )

    args = parser.parse_args()

    main(args)
