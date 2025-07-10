import mplhep as hep
import matplotlib.pyplot as plt
import numpy as np
import ROOT

from rich.console import Console

from .sample import construct_data_samples
from .config import Configuration
from .definitions import add_all_values

console = Console()

def make_axo_style_score_plot(
        sample,
        score,
        score_name,
        working_point,
        output_path,
        x_axis_bounds=(0, 180.0),
        x_axis_label='Emulated CICADA Score',
        working_point_label = 'CICADA Nominal',
        pure_label = 'CICADA Pure',
):
    hep.style.use("CMS")
    hep.cms.text("Preliminary", loc=2)

    values = sample.df.AsNumpy(
        [
            score,
            'pure_event'
        ]
    )

    scores = values[score]
    pure = values['pure_event']

    working_point_scores = scores[scores > working_point]
    working_point_pures = pure[scores > working_point]
    pure_scores = working_point_scores[working_point_pures == 1]

    overall_hist = np.histogram(
        scores,
        bins=90,
        range=x_axis_bounds
    )
    working_point_hist = np.histogram(
        working_point_scores,
        bins=90,
        range=x_axis_bounds,
    )
    pure_hist = np.histogram(
        pure_scores,
        bins=90,
        range=x_axis_bounds
    )

    overall_fig = hep.histplot(
        overall_hist,
        label='All Zero Bias'
    )
    working_point_fig = hep.histplot(
        working_point_hist,
        label=working_point_label,
    )
    pure_score_fig = hep.histplot(
        pure_hist,
        label=pure_label,
        linestyle='--',
    )

    plt.legend(loc='upper right', title='Zero Bias Triggered Events')
    plt.xlabel(x_axis_label)
    plt.ylabel('Events')
    plt.yscale('log')
    plt.ylim(1.0, np.max(overall_hist[0])*100.0)

    hist_name = f'{score_name}_axo_style_score_plot'

    plt.savefig(
        f'{output_path}/{hist_name}.png'
    )
    plt.savefig(
        f'{output_path}/{hist_name}.pdf'
    )
    plt.close()
