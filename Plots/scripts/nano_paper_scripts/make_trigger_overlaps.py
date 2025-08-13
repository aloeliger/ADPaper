import numpy as np
import ROOT

import src.trigger_overlaps as trigger_overlaps
from src.sample import construct_data_samples, construct_mc_samples
from src.config import Configuration
import src.definitions as definitions

from rich.console import Console
from pathlib import Path
import argparse

console = Console()

def main(args):
    console.log('Making trigger overlap plots for samples')

    config = Configuration.GetConfiguration().configs

    output_path = Path(config['output path']+'/trigger_overlaps/')
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

    #Data overlaps
    trigger_overlaps.make_total_trigger_overlap_plot(
        sample=data_sample,
        sample_name='RunI',
        output_path=output_path,
    )
    trigger_overlaps.sample_leading_trigger_overlap_plot(
        sample=data_sample,
        sample_name='RunI',
        output_path=output_path
    )

    #MC overlaps
    for mc_name in mc_samples:
        trigger_overlaps.make_total_trigger_overlap_plot(
            sample=mc_samples[mc_name],
            sample_name=mc_name,
            output_path=output_path
        )
        trigger_overlaps.sample_leading_trigger_overlap_plot(
            sample=mc_samples[mc_name],
            sample_name=mc_name,
            output_path=output_path,
        )

    console.log('Done making trigger overlap plots for samples')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Make trigger overlap plots")
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Operate on a smaller set, to make checking changes easier',
    )

    args = parser.parse_args()

    main(args)
