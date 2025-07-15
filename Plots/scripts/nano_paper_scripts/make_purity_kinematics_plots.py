import mplhep as hep
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
import ROOT

import src.purity_kinematics as purity_kinematics
from src.sample import construct_data_samples
from src.config import Configuration
import src.definitions as definitions

from rich.console import Console
from pathlib import Path
import argparse

console = Console()

def main(args):
    console.log('Making trigger pure kinematics plots')

    config = Configuration.GetConfiguration().configs

    output_path = Path(config['output path']+'/purity_kinematics_plots/')
    output_path.mkdir(exist_ok=True, parents=True)

    data_sample = construct_data_samples()['RunI_SpecificRuns']

    definitions.add_all_values(data_sample)
    definitions.make_collisions_runs_cuts(data_sample)
    definitions.add_HLT_and_scouting_values(data_sample)

    purity_kinematics.make_L1HT_purity_plot(
        sample = data_sample,
        sample_name = 'RunI',
        score_name = 'CICADA2024',
        score_display_name='CICADA 2024',
        score_label = 'CICADA Medium',
        score_value = config['CICADA working points']['CICADA2024']['CICADA Medium'],
        output_dir = output_path
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Make pure event trigger kinematics')

    args = parser.parse_args()

    main(args)
