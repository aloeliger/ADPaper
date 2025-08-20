import ROOT
import os
import numpy as np
from pathlib import Path

from rich.console import Console
from rich.progress import track
from rich.live import Live
from pathlib import Path
import uproot
import mplhep as hep
import matplotlib.pyplot as plt

from sklearn.gaussian_process import GaussianProcessRegressor
import sklearn.gaussian_process as gp
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

from time import perf_counter

import pickle as pkl
import h5py

import argparse

console = Console()
nprng = np.random.default_rng(seed=42)

def get_inputs(list_of_files):
    file_and_tree = [x+':Events' for x in list_of_files]
    console.log("Getting Total Number of Entries...")
    the_chain = ROOT.TChain("Events")
    for fileName in list_of_files:
        the_chain.Add(fileName)
    total_entries = the_chain.GetEntries()
    console.log(f'Done! {total_entries}')
    #batch_num = 0
    branches_to_load = [
        "CICADA2025_CICADAScore",
        "axol1tl_v5_AXOScore"
    ]

    cicada_scores = []
    axo_scores = []

    used_branches = lambda x: x in branches_to_load
    #for batch in track(uproot.iterate(file_and_tree, filter_name=used_branches), console=console):
    console.log('Load')
    with Live(console=console) as live:
        batch_num = 0
        total_scores = 0
        start_time = perf_counter()
        for batch in uproot.iterate(file_and_tree, filter_name=used_branches, step_size="500 MB"):
            cicada_scores += list(batch.CICADA2025_CICADAScore)
            axo_scores += list(batch.axol1tl_v5_AXOScore)
            total_scores += len(batch.CICADA2025_CICADAScore)
            batch_num+=1
            current_time = perf_counter()
            live.update(f'Batch #: {batch_num}\nTotal scores: {total_scores}/{total_entries} {total_scores/total_entries:.1%}\nTime Elapsed: {current_time-start_time:.4g}s\nAverage Interval Time:{(current_time-start_time)/batch_num:.4g}s')
            
        
    scores = np.array(cicada_scores)
    axo_scores = np.array(axo_scores)
    console.log('Done!')
    return cicada_scores, axo_scores

def convert_eff_to_rate(eff, nBunches = 2544):
    return eff * (float(nBunches) * 11425e-3)

def convert_rate_to_eff(rate, nBunches=2544):
    return rate / (float(nBunches) * 11425e-3)

def make_overlap_rates(
        cicada_scores,
        axo_scores,
):
    rates = []
    combined_rates = []
    combined_rate_uncerts = []

    desired_rates = np.geomspace(1e-2, 1e1, 50)
    desired_effs = convert_rate_to_eff(desired_rates)

    cicada_scores = np.array(cicada_scores)
    axo_scores = np.array(axo_scores)
    combined_scores = np.stack((cicada_scores, axo_scores), axis=1)
    console.log('Bootstrapping')
    new_samples = [
        combined_scores[nprng.choice(combined_scores.shape[0], size=len(combined_scores), replace=True)]
        for _ in range(100)
    ]
    new_samples = np.array(new_samples)

    desired_cicada_effs = desired_effs
    desired_axo_effs = desired_effs

    desired_cicada_percentiles = 100.0*(1.0-desired_cicada_effs)
    desired_axo_percentiles = 100.0*(1.0-desired_axo_effs)

    console.log('Making thresholds')
    cicada_thresholds = np.percentile(new_samples[:, :, 0], desired_cicada_percentiles, axis=1)
    cicada_thresholds = np.swapaxes(cicada_thresholds, 0, 1)
    axo_thresholds = np.percentile(new_samples[:, :, 1], desired_axo_percentiles, axis=1)
    axo_thresholds = np.swapaxes(axo_thresholds, 0, 1)

    console.log('Making mask')
    #cicada_mask = new_samples[:, :, None, 0] > cicada_thresholds[:, None, :] #Final shape (n_samples, n_score, n_grid)
    # new_samples [n_samples, n_score, axo or cicada]
    #axo_mask = new_samples[:, :, None, 1] > axo_thresholds[:, None, :]
    combined_mask = (new_samples[:, :, None, 0] > cicada_thresholds[:, None, :])  | (new_samples[:, :, None, 1] > axo_thresholds[:, None, :])
    #console.log(combined_mask.shape)

    #Average over the scores axis to get per threshold efficiencies
    console.log('Combined effs')
    combined_effs = np.mean(combined_mask, axis=1)
    all_combined_rates = convert_eff_to_rate(combined_effs)
    cicada_rates = convert_eff_to_rate(desired_cicada_effs)
    axo_rates = convert_eff_to_rate(desired_axo_effs)
    combined_rates = np.mean(all_combined_rates, axis=0)
    combined_rate_uncerts = np.std(all_combined_rates, axis=0) #std nor var here
    
    console.log('Done!')

    return desired_rates, combined_rates, combined_rate_uncerts

def make_model(
        cicada_scores,
        axo_scores,
        output_dir,
):
    desired_rates = np.geomspace(1e-2, 1e1, 7)
    desired_effs = convert_rate_to_eff(desired_rates)

    cicada_scores = np.array(cicada_scores)
    axo_scores = np.array(axo_scores)
    combined_scores = np.stack((cicada_scores, axo_scores), axis=1)
    console.log('Bootstrapping')
    new_samples = [
        combined_scores[nprng.choice(combined_scores.shape[0], size=len(combined_scores), replace=True)]
        for _ in range(100)
    ]
    new_samples = np.array(new_samples)
    
    desired_cicada_effs, desired_axo_effs = np.meshgrid(
        desired_effs,
        desired_effs,
    )
    desired_cicada_effs = desired_cicada_effs.ravel() #shape (n_grid)
    desired_axo_effs = desired_axo_effs.ravel()

    # stacked_desired_effs = np.stack((desired_cicada_effs, desired_axo_effs), axis=1)
    # subset_desired_effs = stacked_desired_effs[nprng.choice(stacked_desired_effs.shape[0], size=60, replace=False)] #take a small subset of points to keep the masking/model fitting feasible
    # desired_cicada_effs=subset_desired_effs[:, 0]
    # desired_axo_effs=subset_desired_effs[:, 1]

    desired_cicada_percentiles = 100.0*(1.0-desired_cicada_effs)
    desired_axo_percentiles = 100.0*(1.0-desired_axo_effs)

    console.log('Making thresholds')
    cicada_thresholds = np.percentile(new_samples[:, :, 0], desired_cicada_percentiles, axis=1)
    cicada_thresholds = np.swapaxes(cicada_thresholds, 0, 1)
    axo_thresholds = np.percentile(new_samples[:, :, 1], desired_axo_percentiles, axis=1)
    axo_thresholds = np.swapaxes(axo_thresholds, 0, 1)

    console.log('Making mask')
    #cicada_mask = new_samples[:, :, None, 0] > cicada_thresholds[:, None, :] #Final shape (n_samples, n_score, n_grid)
    # new_samples [n_samples, n_score, axo or cicada]
    #axo_mask = new_samples[:, :, None, 1] > axo_thresholds[:, None, :]
    combined_mask = (new_samples[:, :, None, 0] > cicada_thresholds[:, None, :])  | (new_samples[:, :, None, 1] > axo_thresholds[:, None, :])
    #console.log(combined_mask.shape)

    #Average over the scores axis to get per threshold efficiencies
    console.log('Combined effs')
    combined_effs = np.mean(combined_mask, axis=1)
    all_combined_rates = convert_eff_to_rate(combined_effs)
    cicada_rates = convert_eff_to_rate(desired_cicada_effs)
    axo_rates = convert_eff_to_rate(desired_axo_effs)
    combined_rates = np.mean(all_combined_rates, axis=0)
    combined_rate_uncerts = np.var(all_combined_rates, axis=0)
    console.log('Done!')    

    kernel = gp.kernels.ConstantKernel(1.0, constant_value_bounds=(1e-6,1e8))*gp.kernels.RBF(1.0, length_scale_bounds=(1e-2, 1e2))#+gp.kernels.WhiteKernel(noise_level_bounds=(1e-6, 1e6))
    x = np.stack((cicada_rates, axo_rates), axis=1)
    y = combined_rates.reshape((-1,1))

    combined_rates_model = Pipeline(
        [
            (
                'scaling',
                StandardScaler()
            ),
            (
                'gpr',
                GaussianProcessRegressor(
                    n_restarts_optimizer=5,
                    kernel=kernel,
                    #normalize_y=True,
                    alpha=combined_rate_uncerts,
                )
            )
        ]
    )
    combined_rates_model.fit(x, y)
    #Let's pickle the resulting model
    with open(f'{output_dir}/combined_rates_model.pkl', 'wb') as theFile:
        pkl.dump(combined_rates_model, theFile)
    with h5py.File(f'{output_dir}/model_data.h5', 'w') as the_file:
        the_file.create_dataset(f'x', data=x)
        the_file.create_dataset(f'y', data=y)
        the_file.create_dataset(f'y_uncert', data=combined_rate_uncerts)

    return combined_rates_model, None, None

def make_overlap_plots(
        rates,
        combined_rates,
        combined_rate_uncerts,
        output_dir,
        debug=False,
        model = None
):
    fig, axes = plt.subplots(2, 1)

    hep.style.use("CMS")
    hep.cms.label("Preliminary", data=True, rlabel='Run 392991', ax=axes[0], fontsize=11)

    if model is not None:
        model_inputs = np.stack((rates, rates), axis=1)
        model_predictions, model_uncerts = model.predict(model_inputs, return_std=True)

        combined_rates_model = model_predictions.reshape((-1,))
        combined_rates_model_uncerts = model_uncerts.reshape((-1,))
        
    
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
    if model is not None:
        axes[0].plot(
            rates,
            combined_rates_model,
            label='Smoothed Model (GP)'
        )
        axes[0].fill_between(
            rates,
            combined_rates_model-combined_rates_model_uncerts,
            combined_rates_model+combined_rates_model_uncerts,
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
    axes[0].set_ylabel('Combined AD Rate (kHz)', fontsize=11)
    axes[0].grid()
    axes[0].legend(fontsize=11)
    
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
    if model is not None:
        ratio_combined_rates_model = combined_rates_model / rates
        ratio_combined_rates_model_uncerts = combined_rates_model_uncerts/rates
        axes[1].plot(
            rates,
            ratio_combined_rates_model,
            label = 'Ratio (Model)',
        )
        axes[1].fill_between(
            rates,
            ratio_combined_rates_model-ratio_combined_rates_model_uncerts,
            ratio_combined_rates_model+ratio_combined_rates_model_uncerts,
            alpha=0.5,
            label=r'$\pm 1 \sigma$'
        )

    axes[1].set_xscale('log')
    #axes[1].set_yscale('log')
    axes[1].set_ylim(bottom=0.5, top=2.5)
    axes[1].set_xlabel('Individual AD Trigger Rate (kHz)', fontsize=11)
    axes[1].set_ylabel('Combined Rate / Perfect Overlap', va='center', ha='center', fontsize=11)
    axes[1].grid()
    axes[1].legend(fontsize=11)

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.0)

    hist_name = 'CICADA_AXO_Overlap'
    if debug:
        hist_name = 'CICADA_AXO_Overlap_debug'
    if model is not None:
        hist_name+='_withModel'
    
    plt.savefig(
        f'{hist_name}.png',
        bbox_inches='tight',
    )

    plt.savefig(
        f'{output_dir}/{hist_name}.pdf',
        bbox_inches='tight',
    )

def main(args):
    console.log('Start')

    files_path = '/hdfs/store/user/aloelige/ZeroBias/Operations_ZeroBias_Run2025C_Run392991_13Aug2025'
    all_files = []
    file_index = 0
    for root, dirs, files in os.walk(files_path):
        for fileName in files:
            file_index+=1
            if args.debug:
                if file_index > 20:
                    continue
            # elif file_index % 2 != 0:
            #     continue 
            the_file = f'{root}/{fileName}'
            all_files.append(the_file)
    cicada_scores, axo_scores = get_inputs(all_files)    
    
    console.log(
        f'# of Scores: {len(cicada_scores)} & {len(axo_scores)}'
    )

    #Let's make an output directory.
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    #Okay, here's how this is going to work
    #First,we are going to specify a certain number of rates between
    # 1Hz and 100 kHz, and figure out what percentile that corresponds to
    # Then for each trigger we figure out what score that is
    # Then for a given rate/score pair, we go through and figure out
    # the combined rate. We can use boot strapping to get an idea what
    # our uncertainty is

    # console.log('Deriving trigger thresholds')

    # # desired_log_rates = np.linspace(np.log(0.01), np.log(10.0), 100)
    # # desired_rates = np.exp(desired_log_rates)
    # #desired_rates = np.append(np.sort(np.linspace([1e-2, 1e-1, 1e0],[1e-1, 1e0, 1e1], 9, endpoint=False).ravel()), [1e1], axis=0)
    # desired_rates = np.geomspace(1e-2, 1e1, 70)
    # desired_effs = convert_rate_to_eff(desired_rates)
    # desired_percentiles = 100.0*(1.0-desired_effs)
    # # console.print(desired_rates)
    # # console.print(desired_percentiles)
    # # exit(0)

    # cicada_rate_scores = {}
    # axo_rate_scores = {}
    # for index in track(range(len(desired_rates)), console=console):
    #     rate = desired_rates[index]
    #     cicada_score = np.percentile(
    #         cicada_scores,
    #         desired_percentiles[index]
    #     )

    #     axo_score = np.percentile(
    #         axo_scores,
    #         desired_percentiles[index]
    #     )

    #     cicada_rate_scores[rate] = cicada_score
    #     axo_rate_scores[rate] = axo_score

    # console.log('CICADA Rate Scores')
    # console.log(cicada_rate_scores)
    # console.log("AXO Rate Scores")
    # console.log(axo_rate_scores)

    console.log('Making combined model')
    model, cicada_rate_model, axo_rate_model = make_model(
        cicada_scores,
        axo_scores,
        #cicada_rate_scores,
        #axo_rate_scores,
        args.output_dir
    )

    
    console.log('Making overlap rates')

    rates, combined_rates, combined_rate_uncerts = make_overlap_rates(
        cicada_scores,
        axo_scores,
        # cicada_rate_scores,
        # axo_rate_scores,
    )

    console.log('Individual Trigger Rates:')
    console.log(rates)
    console.log()
    console.log('Derived Combined Rates:')
    console.log(combined_rates)
    console.log()
    console.log('Combined rate uncerts:')
    console.log(combined_rate_uncerts)
    console.log()

    console.log('Making overlap plots')

    make_overlap_plots(
        rates,
        combined_rates,
        combined_rate_uncerts,
        output_dir=output_dir,
        debug=args.debug,
    )

    make_overlap_plots(
        rates,
        combined_rates,
        combined_rate_uncerts,
        output_dir=output_dir,
        debug=args.debug,
        model=model,
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
    parser.add_argument(
        '--output_dir',
        default='./test',
        help='Location to store resulting plots and models'
    )

    args = parser.parse_args()
    
    main(args)
