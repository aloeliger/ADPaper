import mplhep as hep
import matplotlib.pyplot as plt
import numpy as np
import ROOT
import itertools
from pathlib import Path
from rich.console import Console

from .sample import construct_data_samples
from .config import Configuration
from .definitions import add_all_values

console = Console()

def make_L1HT_purity_plot(
        sample,
        sample_name,
        score_name,
        score_display_name,
        score_label,
        score_value,
        output_dir
):
    hep.style.use('CMS')
    hep.cms.text(f'Preliminary', loc=2)

    triggered_events_df = sample.df.Filter(f'{score_name} > {score_value}')
    l1_pure_events = triggered_events_df.Filter('pure_event == 1')
    hlt_pure_events = triggered_events_df.Filter('pure_hlt_event == 1')
    scouting_pure_events = triggered_events_df.Filter('pure_scouting_event == 1')
    
    L1_HT = triggered_events_df.AsNumpy(['L1_HT'])['L1_HT']
    l1_pure_L1_HT = l1_pure_events.AsNumpy(['L1_HT'])['L1_HT']
    hlt_pure_L1_HT = hlt_pure_events.AsNumpy(['L1_HT'])['L1_HT']
    scouting_pure_L1_HT = scouting_pure_events.AsNumpy(['L1_HT'])['L1_HT']

    fig_total = hep.histplot(
        np.histogram(
            L1_HT,
            bins=20,
            range=(0.0, 2000.0),
        ),
        label = f'{score_label}'
    )

    fig_l1 = hep.histplot(
        np.histogram(
            l1_pure_L1_HT,
            bins=20,
            range=(0.0, 2000.0),
        ),
        label=f'{score_label} pure w.r.t L1'
    )

    fig_hlt = hep.histplot(
        np.histogram(
            hlt_pure_L1_HT,
            bins=20,
            range=(0.0, 2000.0),
        ),
        label=f'{score_label} pure w.r.t Full Reco'
    )

    fig_scouting = hep.histplot(
        np.histogram(
            scouting_pure_L1_HT,
            bins=20,
            range=(0.0, 2000.0)
        ),
        label=f'{score_label} pure w.r.t Scouting'
    )

    plt.xlabel('L1 H_{T}')
    plt.ylabel('Events')

    hist_name = f'{sample_name}_{score_name}_purity_L1_HT'

    plt.savefig(
        f'{output_dir}/{hist_name}.png'
    )
    plt.savefig(
        f'{output_dir}/{hist_name}.pdf'
    )
    plt.close()
