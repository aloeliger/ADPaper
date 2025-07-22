import mplhep as hep
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
import ROOT
import itertools
from pathlib import Path
from rich.console import Console

from .sample import construct_data_samples
from .config import Configuration

console = Console()

def make_leading_object_scatter(
        sample,
        sample_name,
        score_name,
        score_display_name,
        object_to_use,
        output_path
):
    #Okay, first thing's first, we need to narrow this down to a dataframe with at least one object in it
    multiplicity_var = f'n{object_to_use}'
    object_df = sample.df.Filter(f'{multiplicity_var} >= 1')

    #Let's define a new column for the leading object pt
    leading_var = f'{object_to_use}_leading_pt'
    object_df = object_df.Define(
        leading_var,
        f'{object_to_use}_pt[0]'
    )

    #Let's also make sure we're dealing with scores that are non zero
    object_df = object_df.Filter(f'{score_name} > 10.0')

    values = object_df.AsNumpy([score_name, leading_var])
    score_values = values[score_name]
    leading_values = values[leading_var]

    hep.style.use("CMS")
    hep.cms.text(f'Preliminary', loc=2)

    fig = hep.hist2dplot(
        np.histogram2d(
            score_values,
            leading_values,
            bins=20,
            density=True
        )
    )

    plt.yscale('log')

    plt.xlabel(f"{score_display_name} (Score > 10)")
    plt.ylabel(f'{object_to_use} Leading Object'+' $p_{T}$')

    hist_name = f'{sample_name}_{score_name}_{object_to_use}_leading_object'
    plt.savefig(
        f'{output_path}/{hist_name}.png'
    )
    plt.savefig(
        f'{output_path}/{hist_name}.pdf'
    )

    plt.close()
