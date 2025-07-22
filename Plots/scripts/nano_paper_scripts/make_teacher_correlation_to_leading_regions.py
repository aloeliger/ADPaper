import numpy as np
import ROOT
import huggingface_hub

import src.teacher_correlation_to_leading_regions as teacher_correlation
from src.sample import construct_data_samples
from src.config import Configuration
import src.definitions as definitions

from rich.console import Console
from pathlib import Path
import argparse

console = Console()

def main(args):
    console.log('Making teacher correlation to leading region ET')

    config = Configuration.GetConfiguration().configs

    output_path = Path(config['output path']+'/teacher_correlation_to_leading_regions/')
    output_path.mkdir(exist_ok=True, parents=True)

    data_sample = construct_data_samples()['RunI']

    if args.debug:
        data_sample.df = data_sample.df.Range(1000)
    else:
        data_sample.df = data_sample.df.Range(7500000)

    definitions.make_collisions_runs_cuts(data_sample)
    definitions.add_all_values(data_sample)

    teacher_model = huggingface_hub.from_pretrained_keras('cicada-project/teacher-v.0.1')

    teacher_correlation.make_leading_region_error_plot(
        sample=data_sample,
        teacher_model = teacher_model,
        output_path=output_path
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Make teacher correlation to leading region ET')

    parser.add_argument(
        '--debug',
        action='store_true',
        help='Operate on a smaller set, to make checking changes easier'
    )

    args = parser.parse_args()

    main(args)
