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

def make_scatter_plot(
        sample,
        sample_name,
        cicada_name,
        axo_name,
        cicada_score,
        axo_score,
        output_path
):    
    values = sample.df.AsNumpy([cicada_score, axo_score])
    cicada_scores = values[cicada_score]
    axo_scores = values[axo_score]

    correlation = np.corrcoef(cicada_scores, axo_scores)[0][1]

    hep.style.use("CMS")
    hep.cms.text(f"Preliminary, Corr = {correlation:.02g}", loc=0)
    
    fig = hep.hist2dplot(
        np.histogram2d(
            cicada_scores,
            axo_scores,
            bins=20,
            density=True
        ),
        norm = colors.LogNorm(),
    )

    plt.xlabel('CICADA Score')
    plt.ylabel('AXO Score')

    hist_name = f'{sample_name}_{cicada_name}_{axo_name}_2d_scatter'
    
    plt.savefig(
        f'{output_path}/{hist_name}.png'
    )

    plt.savefig(
        f'{output_path}/{hist_name}.pdf'
    )

    plt.close()

    #
    # Reduce the range
    #

    hep.style.use("CMS")
    hep.cms.text(f"Preliminary, Corr = {correlation:.02g}", loc=0)

    if np.sum((cicada_scores > 5.0)) > 0 and np.sum((axo_scores>100.0)) > 0:
    
        fig = hep.hist2dplot(
            np.histogram2d(
                cicada_scores,
                axo_scores,
                bins=20,
                density=True,
                range=[
                    [
                        5.0,
                        max(6.0, np.max(cicada_scores)),
                    ],
                    [
                        100.0,
                        max(100.0, np.max(axo_scores))
                    ]
                ]
            ),
            norm = colors.LogNorm()
        )

        plt.xlabel('CICADA Score')
        plt.ylabel('AXO Score')
        
        hist_name = f'{cicada_name}_{axo_name}_2d_scatter_reduced'
        
        plt.savefig(
            f'{output_path}/{hist_name}.png'
        )
        
        plt.savefig(
            f'{output_path}/{hist_name}.pdf'
        )

        plt.close()
