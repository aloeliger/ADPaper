import mplhep as hep
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
import ROOT

import src.signal_confusion_plots as signal_confusion_plots
from src.sample import construct_data_samples, construct_mc_samples
from src.config import Configuration
import src.definitions as definitions

from rich.console import Console
from pathlib import Path
import argparse

console = Console()

def main(args):
    console.log('Making signal confusion plots')

    current_config = Configuration.GetConfiguration().configs

    output_path = Path(current_config['output path']+'/signal_confusion_plots/')
    output_path.mkdir(exist_ok=True, parents=True)

    data_sample = construct_data_samples()['RunI']
    mc_samples = construct_data_samples()

    definitions.add_all_values(data_sample)
    for mc_name in mc_samples:
        definitions.add_all_values(mc_samples[mc_name], is_mc=True)

    cicada_names = current_config['CICADA Scores']
    axo_names = current_config['AXO Scores']

    for cicada_name in cicada_names:
        signal_confusion_plots.make_confusion_plot(
            sample = data_sample,
            sample_name = 'RunI',
            score_display_name = cicada_name,
            score_name = cicada_names[cicada_name],
            score_value = current_config['CICADA working points'][cicada_name]['CICADA Medium'],
            output_dir=output_path
        )

    for axo_name in axo_names:
        signal_confusion_plots.make_confusion_plot(
            sample = data_sample,
            sample_name = 'RunI',
            score_display_name = axo_name,
            score_name = axo_names[axo_name],
            score_value = current_config['AXO working points'][axo_name]['Nominal'],
            output_dir=output_path
        )

    for mc_sample_name in mc_samples:
        mc_sample = mc_samples[mc_sample_name]
        for cicada_name in cicada_names:
            signal_confusion_plots.make_confusion_plot(
                sample = mc_sample,
                sample_name = mc_sample_name,
                score_display_name = cicada_name,
                score_name = cicada_names[cicada_name],
                score_value = current_config['CICADA working points'][cicada_name]['CICADA Medium'],
                output_dir=output_path
            )
            
        for axo_name in axo_names:
            signal_confusion_plots.make_confusion_plot(
                sample = mc_sample,
                sample_name = mc_sample_name,
                score_display_name = axo_name,
                score_name = axo_names[axo_name],
                score_value = current_config['AXO working points'][axo_name]['Nominal'],
                output_dir=output_path
            )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Make signal confusion plots")

    args = parser.parse_args()

    main(args)
