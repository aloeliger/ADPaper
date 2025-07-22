import mplhep as hep
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
import ROOT

from pathlib import Path
from rich.console import Console

from .sample import construct_data_samples
from .sample import construct_mc_samples

console = Console()

def make_confusion_plot(
        sample,
        sample_name,
        score_display_name,
        score_name,
        score_value,
        output_dir,
):
    uncaught_cut = f'{score_name} < {score_value} && !l1_event'
    AD_only_cut = f'{score_name} > {score_value} && pure_event'
    trigger_only_event = f'{score_name} < {score_value} && l1_event'
    AD_and_trigger_event = f'{score_name} > {score_value} && l1_event'

    all_events = sample.df.Count()
    uncaught_events = sample.df.Filter(uncaught_cut).Count()
    AD_only_events = sample.df.Filter(AD_only_cut).Count()
    trigger_only_events = sample.df.Filter(trigger_only_event).Count()
    AD_and_trigger_events = sample.df.Filter(AD_and_trigger_event).Count()

    uncaught_fraction = uncaught_events.GetValue() / all_events.GetValue()
    AD_only_fraction = AD_only_events.GetValue() / all_events.GetValue()
    trigger_only_fraction = trigger_only_events.GetValue() / all_events.GetValue()
    AD_and_trigger_fraction = AD_and_trigger_events.GetValue() / all_events.GetValue()

    fig, ax = plt.subplots()

    hep.style.use("CMS")
    hep.cms.text(f'Preliminary', loc=2)
    
    hep.hist2dplot(
        np.histogram2d(
            [0,0,1,1],
            [0,1,0,1],
            bins=2,
            weights=[uncaught_fraction, AD_only_fraction, trigger_only_fraction, AD_and_trigger_fraction]
        ),
        ax=ax
    )

    ax.text(0.25, 0.25, uncaught_fraction, color='white', ha='center', va='center',fontsize=16)
    ax.text(0.25, 0.75, AD_only_fraction, color='white', ha='center', va='center',fontsize=16)
    ax.text(0.75, 0.25, trigger_only_fraction, color='white', ha='center', va='center',fontsize=16)
    ax.text(0.75, 0.75, AD_and_trigger_fraction, color='white', ha='center', va='center',fontsize=16)

    bin_labels = ["False", "True"]
    ax.set_xticks([0.25, 0.75])
    ax.set_yticks([0.25, 0.75])
    ax.set_xticklabels(bin_labels)
    ax.set_yticklabels(bin_labels)
    
    #print(fig)

    plt.xlabel('L1 Trigger')
    plt.ylabel(f'{score_display_name}')

    score_str = str(score_value).replace('.', 'p')
    hist_name = f'{sample_name}_{score_name}_{score_str}_confusion'
    
    plt.savefig(
        f'{output_dir}/{hist_name}.png'
    )
    plt.savefig(
        f'{output_dir}/{hist_name}.pdf'
    )
    plt.close()
