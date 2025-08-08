import mplhep as hep
import matplotlib.pyplot as plt
import numpy as np
import ROOT

from rich.console import Console
from rich.table import Table
from .config import Configuration

console = Console()

def make_total_trigger_overlap_plot(
        sample,
        sample_name,
        output_path
):
    hep.style.use("CMS")
    hep.cms.text(f"Preliminary", loc=2)
    
    pure_event_all_count = sample.df.Filter('pure_event_all == 1').Count()
    pure_event_count = sample.df.Filter('pure_event == 1').Count()
    impure_event_count = sample.df.Filter('pure_event == 0').Count()
    impure_event_all_count = sample.df.Filter('pure_event_all == 0').Count()

    fractions = np.array([
        pure_event_all_count.GetValue(),
        pure_event_count.GetValue(),
        impure_event_count.GetValue(),
        impure_event_all_count.GetValue(),
    ])
    
    hist_name = f'{sample_name}_L1T_Overlap_Plot'

    labels = ['Pure W.R.T. all L1T', 'Pure W.R.T. unprescaled triggers', 'Impure W.R.T. unprescaled triggers', 'impure w.r.t all L1T']
    bins = np.array([1,2,3,4])
    fig = hep.histplot(
        np.histogram(
            bins,
            density = True,
            weights = fractions,
        )
    )

    tick_positions = (bins + bins+1) / 2
    plt.xticks(tick_positions, labels)
    plt.ylabel('A.U.')
    plt.tight_layout()
    plt.savefig(f'{output_path}/{hist_name}.png')
    plt.savefig(f'{output_path}/{hist_name}.pdf')

def sample_leading_trigger_overlap_plot(
        sample,
        sample_name,
        output_path,
        n_top_triggers=10,
):
    hep.style.use("CMS")
    hep.cms.text(f"Preliminary", loc=2)

    current_config = Configuration.GetConfiguration().configs
    all_trigger_list = current_config['All_L1_Triggers']

    trigger_counts = {}
    for trigger_name in all_trigger_list:
        trigger_counts[trigger_name] = sample.df.Filter(f'{trigger_name} == 1').Count()

    #process the dict into a sortable thing so we can get the top
    #triggers
    sortable_trigger_counts = []
    for trigger_name in trigger_counts:
        sortable_trigger_counts.append(
            (trigger_name, trigger_counts[trigger_name].GetValue())
        )
    sortable_trigger_counts.sort(key=lambda x: x[1]) #sort on trigger counts
    top_triggers = sortable_trigger_counts[-20:]

    top_trigger_names = [x[0] for x in top_triggers]
    top_trigger_counts = [x[1] for x in top_triggers]
    bins=np.arange(len(top_trigger_names))

    fig = hep.histplot(
        np.histogram(
            bins,
            density=True,
            weights=top_trigger_counts
        )
    )

    tick_positions = (bins+bins+1)/2
    plt.xtikcs(tick_positions, labels)
    plt.ylabel('A.U.')
    plt.tight_layout()
    hist_name = f'{sample_name}_Individual_Trigger_Overlap_Plot'
    plt.savefig(f'{output_path}/{hist_name}.png')
    plt.savefig(f'{output_path}/{hist_name}.pdf')
    
