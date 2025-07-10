import mplhep as hep
import matplotlib.pyplot as plt
import numpy as np
import ROOT

from rich.console import Console
from rich.table import Table

from .sample import construct_data_samples
from .config import Configuration
from .definitions import add_all_values

console = Console()

def build_emulated_overlap_table(
        sample,
        score,
        score_name,
        working_point,
        list_of_triggers,
):
    console.log(f'Score Name: {score_name}, WP: {working_point}')

    #For each trigger in the list, get the number of joint events
    overlap_counts = {}
    counts_df = sample.df.Filter(f'{score} > {working_point}')
    total_counts = counts_df.Count()

    for trigger in list_of_triggers:
        overlap_counts[trigger] = counts_df.Filter(f'{trigger} == 1').Count()
    overlap_percentages = []
    for trigger in list_of_triggers:
        overlap_percentages.append(
            (
                trigger,
                (overlap_counts[trigger].GetValue()/total_counts.GetValue()) * 100.0
            )
        )
    overlap_percentages.sort(key=lambda x: x[1], reverse=True)

    return overlap_percentages

def make_emulated_max_overlap_table_plot(
        overlap_table,
        output_path,
        hist_name,
        n_triggers_to_use=20,
):
    triggers = [x[0] for x in overlap_table]
    overlap = [x[1] for x in overlap_table]
    triggers = triggers[:n_triggers_to_use]
    overlap = overlap[:n_triggers_to_use]

    bars = plt.barh(
        triggers,
        overlap,
    )
    plt.bar_label(bars)
    plt.subplots_adjust(left=0.5)
    hep.cms.label(
        label='Preliminary, Emulated',
        data=True,
        year='2024',
        lumi='Run I'
    )

    plt.savefig(
        f'{output_path}/{hist_name}.png'
    )
    plt.savefig(
        f'{output_path}/{hist_name}.pdf'
    )
    plt.close()

def print_overlap_table(overlap_table, n_triggers=10):
    info_table = Table()
    info_table.add_column('Trigger', justify='left')
    info_table.add_column('% Overlap', justify='right')
    for trigger, overlap in overlap_table[:min(len(overlap_table), n_triggers)]:
        info_table.add_row(
            trigger,
            f'{overlap:.2f}',
        )
    console.print(info_table)
