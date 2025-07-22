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
    mc_samples = construct_mc_samples()

    if args.debug:
        data_sample.df = data_sample.df.Range(1000)
        for sample in mc_samples:
            mc_samples[sample].df = mc_samples[sample].df.Range(1000)

    definitions.make_collisions_runs_cuts(data_sample)
    definitions.add_all_values(data_sample)
    for mc_name in mc_samples:
        definitions.add_all_values(mc_samples[mc_name])

    cicada_names = current_config['CICADA Scores']
    axo_names = current_config['AXO Scores']

    # console.log('Signal confusion: CICADA - Data')
    # signal_confusion_plots.make_confusion_plot(
    #     sample = data_sample,
    #     sample_name = 'RunI',
    #     score_display_name = 'CICADA2024',
    #     score_name = 'CICADA2024_CICADAScore',
    #     score_value = current_config['CICADA working points']['CICADA2024']['CICADA Medium'],
    #     output_dir=output_path
    # )
    
    # #100 kHz benchmark
    # signal_confusion_plots.make_confusion_plot(
    #     sample = data_sample,
    #     sample_name = 'RunI',
    #     score_display_name = 'CICADA2024',
    #     score_name = 'CICADA2024_CICADAScore',
    #     score_value = 85.65,
    #     output_dir=output_path
    # )
    # #70 kHz benchmark
    # signal_confusion_plots.make_confusion_plot(
    #     sample = data_sample,
    #     sample_name = 'RunI',
    #     score_display_name = 'CICADA2024',
    #     score_name = 'CICADA2024_CICADAScore',
    #     score_value = 88.15,
    #     output_dir=output_path
    # )

    # console.log('Signal confusion: AXO - Data')
    # signal_confusion_plots.make_confusion_plot(
    #     sample = data_sample,
    #     sample_name = 'RunI',
    #     score_display_name = 'AXOv4',
    #     score_name = axo_names['AXOv4'],
    #     score_value = current_config['AXO working points']['AXOv4']['Nominal'],
    #     output_dir=output_path
    # )
    # #100 kHz
    # signal_confusion_plots.make_confusion_plot(
    #     sample = data_sample,
    #     sample_name = 'RunI',
    #     score_display_name = 'AXOv4',
    #     score_name = axo_names['AXOv4'],
    #     score_value = 146.0,
    #     output_dir=output_path
    # )
    # #70 kHz benchmark
    # signal_confusion_plots.make_confusion_plot(
    #     sample = data_sample,
    #     sample_name = 'RunI',
    #     score_display_name = 'AXOv4',
    #     score_name = axo_names['AXOv4'],
    #     score_value = 162.0,
    #     output_dir=output_path
    # )

    for mc_sample_name in mc_samples:
        console.log(f'Signal confusion: CICADA - {mc_sample_name}')
        mc_sample = mc_samples[mc_sample_name]
        signal_confusion_plots.make_confusion_plot(
            sample = mc_sample,
            sample_name = mc_sample_name,
            score_display_name = "CICADA2024",
            score_name = "CICADA2024_CICADAScore",
            score_value = current_config['CICADA working points']['CICADA2024']['CICADA Medium'],
            output_dir=output_path
        )

        #100 kHz Benchmark
        signal_confusion_plots.make_confusion_plot(
            sample = mc_sample,
            sample_name = mc_sample_name,
            score_display_name = "CICADA2024",
            score_name = "CICADA2024_CICADAScore",
            score_value = 85.65,
            output_dir=output_path
        )

        #70 kHz Benchmark
        signal_confusion_plots.make_confusion_plot(
            sample = mc_sample,
            sample_name = mc_sample_name,
            score_display_name = "CICADA2024",
            score_name = "CICADA2024_CICADAScore",
            score_value = 88.15,
            output_dir=output_path
        )

        console.log(f'Signal confusion: AXO - {mc_sample_name}')
        signal_confusion_plots.make_confusion_plot(
            sample = mc_sample,
            sample_name = mc_sample_name,
            score_display_name = 'AXOv4',
            score_name = 'axol1tl_v4_AXOScore',
            score_value = current_config['AXO working points']['AXOv4']['Nominal'],
            output_dir=output_path
        )
        
        #100 kHz Benchmark
        signal_confusion_plots.make_confusion_plot(
            sample = mc_sample,
            sample_name = mc_sample_name,
            score_display_name = 'AXOv4',
            score_name = 'axol1tl_v4_AXOScore',
            score_value = 146.0,
            output_dir=output_path
        )

        #70 kHz Benchmark
        signal_confusion_plots.make_confusion_plot(
            sample = mc_sample,
            sample_name = mc_sample_name,
            score_display_name = 'AXOv4',
            score_name = 'axol1tl_v4_AXOScore',
            score_value = 162.0,
            output_dir=output_path
        )
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Make signal confusion plots")
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Operate on a smaller set, to make checking changes easier',
    )

    args = parser.parse_args()

    main(args)
