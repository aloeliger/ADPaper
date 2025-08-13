import h5py
import requests
import tsgauth.oidcauth
import argparse
import time
import numpy as np

from rich.console import Console
from rich.progress import track
from pathlib import Path

console = Console()

class CICADA_seed_info():
    def __init__(self, name, threshold):
        self.name = name
        self.threshold = threshold
        self.threshold_pu_rate_data = []

def main(args):
    console.log('Start!')
    auth = tsgauth.oidcauth.DeviceAuth("cmsoms-prod-public",target_client_id="cmsoms-prod",use_auth_file=True)

    output_path = Path(args.output_path)
    output_path.mkdir(exist_ok=True, parents=True)
    cicada_thresholds = [
        CICADA_seed_info('L1_CICADA_VLoose', 50.0),
        CICADA_seed_info('L1_CICADA_Loose', 60.0),
        CICADA_seed_info('L1_CICADA_Medium', 70.0),
        CICADA_seed_info('L1_CICADA_Tight', 80.0),
        CICADA_seed_info('L1_CICADA_VTight', 90.0),
        CICADA_seed_info('L1_CICADA_VVTight', 110.0),
        CICADA_seed_info('L1_CICADA_VVVTight', 125.0),
        CICADA_seed_info('L1_CICADA_VVVVTight', 135.0),
    ]

    lumi_section_limit = args.lumi_section_limit

    for cicada_threshold in cicada_thresholds:
        console.log(f'Processing: {cicada_threshold.name}')
        
        rate_url = f"https://cmsoms.cern.ch/agg/api/v1/l1algorithmtriggers?fields=name,pre_dt_before_prescale_rate&filter[name][EQ]={cicada_threshold.name}&filter[run_number][EQ]={args.run}&group[granularity]=lumisection&page[limit]={lumi_section_limit}"
        time.sleep(30)
        rate_response = requests.get(rate_url, **auth.authparams(), verify=False)
        rate_response.raise_for_status()
        rate_json = rate_response.json()
        lumi_url = f"https://cmsoms.cern.ch/agg/api/v1/lumisections?filter[run_number][EQ]={args.run}&page[limit]=10000"
        time.sleep(30)
        lumi_response = requests.get(lumi_url, **auth.authparams(), verify=False)
        lumi_response.raise_for_status()
        lumi_json = lumi_response.json()

        rate_blocks = rate_json['data']
        lumi_blocks = lumi_json['data']

        console.log(f'Lumi sections: {len(rate_blocks)}')
        
        for index, rate_block in track(enumerate(rate_blocks), description=f'Processing {cicada_threshold.name}', console=console):
            lumi_block = lumi_blocks[index]

            pileup = float(lumi_block['attributes']['pileup'])
            rate = float(rate_block['attributes']['pre_dt_before_prescale_rate'])
            threshold = cicada_threshold.threshold

            cicada_threshold.threshold_pu_rate_data.append([threshold, pileup, rate])
    with h5py.File(f'{output_path}/threshold_pu_rate_training.h5', 'w') as the_file:
        for cicada_threshold in cicada_thresholds:
            console.log(cicada_threshold)
            the_data = np.array(cicada_threshold.threshold_pu_rate_data)
            console.log(the_data.shape)
            the_file.create_dataset(f'{cicada_threshold.name}', data=the_data)
    console.log('Done!')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Make data to make a rate/threshold/pileup model from')

    parser.add_argument(
        'run',
        type=int,
        nargs='?',
        help='Run to make a model for'
    )

    parser.add_argument(
        '--output_path',
        nargs='?',
        default='./',
        help='Place to store output'
    )

    parser.add_argument(
        '--lumi_section_limit',
        type=int,
        nargs='?',
        default=10000
    )

    args=parser.parse_args()
    main(args)
