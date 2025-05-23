import mplhep as hep
import matplotlib.pyplot as plt
import numpy as np
import ROOT
import itertools
from pathlib import Path
from rich.console import Console
from sklearn.metrics import roc_curve

from .sample import construct_data_samples, construct_mc_samples
from .config import Configuration
from .utilities import convert_eff_to_rate

console = Console()

def make_pure_roc_plot(
        signal_samples,
        signal_sample_names,
        background_sample,
        background_sample_name,
        score_name,
        score_ntuple_name,
        output_path,
):
    hep.style.use("CMS")
    hep.cms.text(f"Preliminary", loc=2)
    
    background_scores = background_sample.df.AsNumpy([score_ntuple_name,])[score_ntuple_name]
    #background_inputs =  background_sample.df.AsNumpy(["SimRegions_et"])["SimRegions_et"]
    #background_inputs = np.array([list(x) for x in background_inputs])
    #background_inputs = np.sum(background_inputs**2, axis=1)

    for index, signal_sample_name in enumerate(signal_sample_names):
        signal_sample = signal_samples[signal_sample_name]
        pure_signal_df = signal_sample.df.Filter('pure_event == 1')
        pure_signal_scores = pure_signal_df.AsNumpy([score_ntuple_name,])[score_ntuple_name]
        #pure_signal_inputs = pure_signal_df.AsNumpy(["SimRegions_et"])["SimRegions_et"]
        #pure_signal_inputs = np.array([list(x) for x in pure_signal_inputs])
        #pure_signal_inputs = np.sum(pure_signal_inputs**2, axis=1)

        y_score = np.append(background_scores, pure_signal_scores, axis=0)
        y_true = np.append(np.zeros(background_scores.shape), np.ones(pure_signal_scores.shape), axis=0)

        #y_inputs = np.append(background_inputs, pure_signal_inputs, axis=0)
        
        fpr, tpr, _ = roc_curve(
            y_true,
            y_score
        )
        rates = convert_eff_to_rate(fpr)

        # input_fpr, input_tpr, _ = roc_curve(
        #     y_true,
        #     y_inputs,
        # )
        # input_rates = convert_eff_to_rate(input_fpr)
        
        plt.plot(
            rates,
            tpr,
            c=f'C{index}',
            label=signal_sample_name,
        )

        # plt.plot(
        #     input_rates,
        #     input_tpr,
        #     c=f'C{index}',
        #     linestyle='--',
        # )

    plt.xlabel('Overall Trigger Rate [kHz]')
    plt.ylabel('Fraction Pure Event Acceptance')

    hist_name = f'{score_name}_pure_rocs'

    plt.xlim(0.2, 100.0)
    plt.ylim(1.0e-3, 4.0)

    plt.xscale('log')
    plt.yscale('log')
    
    plt.legend(loc="upper center")
    
    plt.savefig(
        f'{output_path}/{hist_name}.png'
    )
    plt.savefig(
        f'{output_path}/{hist_name}.pdf'
    )

    plt.close()
