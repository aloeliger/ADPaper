import ROOT
import os
import numpy as np

from rich.console import Console
from rich.progress import track
from pathlib import Path
import uproot
import mplhep as hep
import matplotlib.pyplot as plt

import argparse

console = Console()

def get_inputs(list_of_files):
    file_and_tree = [x+':Events' for x in list_of_files]
    #batch_num = 0
    branches_to_load = [
        "CICADA2025_CICADAScore",
        "axol1tl_v4_AXOScore"
    ]

    cicada_scores = []
    axo_scores = []

    used_branches = lambda x: x in branches_to_load
    for batch in track(uproot.iterate(file_and_tree, filter_name=used_branches), console=console):
        cicada_scores += list(batch.CICADA2025_CICADAScore)
        axo_scores += list(batch.axol1tl_v4_AXOScore)
        
    scores = np.array(cicada_scores)
    pure_scores = np.array(axo_scores)
    return cicada_scores, axo_scores

def convert_eff_to_rate(eff, nBunches = 2544):
    return eff * (float(nBunches) * 11425e-3)

def convert_rate_to_eff(rate, nBunches=2544):
    return rate / (float(nBunches) * 11425e-3)

def make_overlap_rates(
        cicada_scores,
        axo_scores,
        cicada_thresholds,
        axo_thresholds        
):
    rates = []
    combined_rates = []
    combined_rate_uncerts = []

    combined_scores = np.stack((cicada_scores, axo_scores), axis=-1)

    console.log('Make overlap rates')
    for rate in track(cicada_thresholds):
        cicada_score = cicada_thresholds[rate]
        axo_score = axo_thresholds[rate]
        # console.log(f'rate: {rate}')
        # console.log(f'cicada score: {cicada_score}')
        # console.log(f'axo_score: {axo_score}')

        #Boot strap uncertainties
        sample_combined_rates = []
        for i in range(500):
            new_sample = combined_scores[np.random.choice(combined_scores.shape[0], size=len(combined_scores), replace=True)]
            cicada_sample = new_sample[:, 0]
            axo_sample = new_sample[:, 1]
            #cicada_sample = np.random.choice(cicada_scores, size=len(cicada_scores), replace=True)
            #axo_sample = np.random.choice(axo_scores, size=len(axo_scores), replace=True)
        
            cicada_mask = cicada_sample > cicada_score
            axo_mask = axo_sample > axo_score
            total_mask = cicada_mask | axo_mask
            and_mask = cicada_mask & axo_mask

            cicada_pass = len(new_sample[cicada_mask])
            axo_pass = len(new_sample[axo_mask])
            both_pass = len(new_sample[and_mask])

            #console.log(f'pass cicada: {cicada_pass}')
            #console.log(f'pass axo: {axo_pass}')
            #console.log(f'pass and: {both_pass}')
            
            num = len(new_sample[total_mask])
            denom = len(new_sample)

            #console.log(f"num:  {num}")
            #console.log(f"denom: {denom}")

            combined_eff = num/denom
            combined_rate = convert_eff_to_rate(combined_eff)
            sample_combined_rates.append(combined_rate)

        #console.log(sample_combined_rates)
        combined_rate = np.mean(sample_combined_rates)
        combined_rate_uncert = np.std(sample_combined_rates)
        
        combined_rates.append(combined_rate)
        combined_rate_uncerts.append(combined_rate_uncert)
        
        # console.log(f'combined rate: {combined_rate}')
        # console.log(f'combined rate uncert: {combined_rate_uncert}')
        rates.append(rate)
    #console.log()
    return np.array(rates), np.array(combined_rates), np.array(combined_rate_uncerts)

def make_overlap_plots(
        rates,
        combined_rates,
        combined_rate_uncerts,
        debug=False,
):
    fig, axes = plt.subplots(2, 1)

    hep.style.use("CMS")
    hep.cms.label("Preliminary", data=True, rlabel='Run 392991', ax=axes[0], fontsize=10)
    
    axes[0].plot(
        rates,
        combined_rates,
        label='Combined CICADA & AXO Rate'
    )
    axes[0].fill_between(
        rates,
        combined_rates-combined_rate_uncerts,
        combined_rates+combined_rate_uncerts,
        alpha=0.5,
        label=r'$\pm 1 \sigma$'
    )
    axes[0].plot(
        rates,
        rates,
        label='Perfect overlap',
        linestyle='--',
        c='gray'
    )

    axes[0].set_xscale('log')
    axes[0].set_yscale('log')
    axes[0].set_ylabel('Combined AD Rate (kHz)', fontsize=9)
    axes[0].grid()
    axes[0].legend(fontsize=10)
    
    ratio_combined_rates = combined_rates / rates
    ratio_combined_rate_uncerts = combined_rate_uncerts / rates

    axes[1].plot(
        rates,
        ratio_combined_rates,
        label='Ratio',
    )
    axes[1].fill_between(
        rates,
        ratio_combined_rates-ratio_combined_rate_uncerts,
        ratio_combined_rates+ratio_combined_rate_uncerts,
        alpha=0.5,
        label=r'$\pm 1 \sigma$'
    )

    axes[1].set_xscale('log')
    #axes[1].set_yscale('log')
    axes[1].set_ylim(bottom=0.5, top=2.5)
    axes[1].set_xlabel('Individual AD Trigger Rate (kHz)', fontsize=10)
    axes[1].set_ylabel('Combined Rate / Perfect Overlap', va='center', ha='center', fontsize=9)
    axes[1].grid()
    axes[1].legend(fontsize=10)

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.0)

    hist_name = 'CICADA_AXO_Overlap'
    if debug:
        hist_name = 'CICADA_AXO_Overlap_debug'
    
    plt.savefig(
        f'{hist_name}.png',
        bbox_inches='tight',
    )

    plt.savefig(
        f'{hist_name}.pdf',
        bbox_inches='tight',
    )

def main(args):
    console.log('Start')

    files_path = '/hdfs/store/user/aloelige/ZeroBias/Operations_ZeroBias_Run2025C_Run392991_05Jun2025'
    all_files = []
    file_index = 0
    for root, dirs, files in os.walk(files_path):
        for fileName in files:
            file_index+=1
            if args.debug:
                if file_index > 10:
                    continue
            # elif file_index % 2 != 0:
            #     continue 
            the_file = f'{root}/{fileName}'
            all_files.append(the_file)
    cicada_scores, axo_scores = get_inputs(all_files)    
    
    console.log(
        f'# of Scores: {len(cicada_scores)} & {len(axo_scores)}'
    )

    #Okay, here's how this is going to work
    #First,we are going to specify a certain number of rates between
    # 1Hz and 100 kHz, and figure out what percentile that corresponds to
    # Then for each trigger we figure out what score that is
    # Then for a given rate/score pair, we go through and figure out
    # the combined rate. We can use boot strapping to get an idea what
    # our uncertainty is

    console.log('Deriving trigger thresholds')

    desired_log_rates = np.linspace(np.log(0.01), np.log(10.0), 100)
    desired_rates = np.exp(desired_log_rates)
    desired_effs = convert_rate_to_eff(desired_rates)
    desired_percentiles = 100.0*(1.0-desired_effs)
    # console.print(desired_rates)
    # console.print(desired_percentiles)
    # exit(0)

    cicada_rate_scores = {}
    axo_rate_scores = {}
    for index in track(range(len(desired_rates)), console=console):
        rate = desired_rates[index]
        cicada_score = np.percentile(
            cicada_scores,
            desired_percentiles[index]
        )

        axo_score = np.percentile(
            axo_scores,
            desired_percentiles[index]
        )

        cicada_rate_scores[rate] = cicada_score
        axo_rate_scores[rate] = axo_score

    # console.log('CICADA Rate Scores')
    # console.log(cicada_rate_scores)
    # console.log("AXO Rate Scores")
    # console.log(axo_rate_scores)
        
    console.log('Making overlap rates')

    rates, combined_rates, combined_rate_uncerts = make_overlap_rates(
        cicada_scores,
        axo_scores,
        cicada_rate_scores,
        axo_rate_scores,
    )

    console.log('Individual Trigger Rates:')
    console.log(rates)
    console.log()
    console.log('Derived Combined Rates:')
    console.log(combined_rates)
    console.log()

    console.log('Making overlap plots')

    make_overlap_plots(
        rates,
        combined_rates,
        combined_rate_uncerts,
        debug=args.debug
    )
    
    
    console.log(
        'Done'
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Make an AXO/CICADA overlap rate plot")

    parser.add_argument(
        '--debug',
        action='store_true',
        help='Make debug plots with fewer files'
    )

    args = parser.parse_args()
    
    main(args)
