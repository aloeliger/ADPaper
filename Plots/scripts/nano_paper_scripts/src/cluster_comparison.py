import mplhep as hep
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
import ROOT

from pathlib import Path
from rich.console import Console

from .sample import construct_data_samples
from .sample import construct_mc_samples
from sklearn.metrics import roc_curve


console = Console()

def make_cluster_comparison_plot(
        background_sample,
        background_sample_name,
        samples,
        score_name,
        score_display_name,
        output_path,
):
    hep.style.use("CMS")
    hep.cms.text(f"Preliminary", loc=2)

    values = background_sample.df.AsNumpy([score_name, "nL1EmulCaloCluster"])
    background_scores = values[score_name]
    background_nclusters = values["nL1EmulCaloCluster"]

    for index, signal_sample_name in enumerate(samples):
        signal_sample = samples[signal_sample_name]
        sample_df = signal_sample.df

        sample_values = sample_df.AsNumpy([score_name, "nL1EmulCaloCluster"])
        sample_scores = sample_values[score_name]
        sample_nclusters = sample_values['nL1EmulCaloCluster']

        y_score = np.append(
            background_scores,
            sample_scores,
            axis=0
        )
        y_cluster_scores = np.append(
            background_nclusters,
            sample_nclusters,
            axis=0
        )

        y_true = np.append(
            np.zeros((len(background_scores),)),
            np.ones((len(sample_scores),)),
            axis=0,
        )

        fpr_score, tpr_score, _ = roc_curve(
            y_true,
            y_score,
        )
        fpr_clusters, tpr_clusters, _ = roc_curve(
            y_true,
            y_cluster_scores,
        )

        plt.plot(
            fpr_score,
            tpr_score,
            c=f'C{index}',
            label=f'{signal_sample_name}'
        )

        plt.plot(
            fpr_clusters,
            tpr_clusters,
            c=f'C{index}',
            linestyle='--',
        )

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
        
    plt.plot(
        [-10.0],
        [-10.0],
        c='gray',
        label=f'{score_display_name}'
    )
    plt.plot(
        [-10.0],
        [-10.0],
        c='gray',
        linestyle='--',
        label='Clusters'
    )
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)

    plt.legend(loc="lower right")
    plt.tight_layout()

    hist_name = f'{score_display_name}_cluster_comparison'
    plt.savefig(
        f'{output_path}/{hist_name}.png'
    )
    plt.savefig(
        f'{output_path}/{hist_name}.pdf'
    )

    plt.close()
