import mplhep as hep
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
import ROOT

from rich.console import Console

from .sample import construct_data_samples
from .config import Configuration
from skimage.measure import block_reduce

console = Console()

def make_eta_losses_plot(
        sample,
        teacher_model,
        output_path
):
    hep.style.use("CMS")
    hep.cms.text(f'Preliminary', loc=2)

    values = sample.df.AsNumpy(['Regions_et','Regions_ieta', 'Regions_iphi'])

    regions_et = values["Regions_et"]
    regions_eta = values["Regions_ieta"]
    regions_phi = values["Regions_iphi"]

    regions_et = np.array([list(x) for x in regions_et])
    regions_eta = np.array([list(x) for x in regions_eta])
    regions_phi = np.array([list(x) for x in regions_phi])
    
    # #Okay, now we need to arrange the regions
    
    regions_et = regions_et.reshape((-1,18,14))
    batch_indices = np.arange(regions_et.shape[0])[:, None]
    #print(regions_et.shape)
    #print(regions_phi.shape)
    #print(regions_eta.shape)
    regions_et = regions_et[batch_indices, regions_phi, regions_eta]
    #print(regions_et.shape)
    regions_et = regions_et.reshape((-1, 18, 14))

    region_predictions=teacher_model.predict(regions_et)
    region_losses = (regions_et.reshape((-1, 18, 14, 1))-region_predictions)**2

    sum_total_losses = np.sum(region_losses, axis=(1,2,3))
    avg_total_losses = np.mean(region_losses, axis=(1,2,3))

    sum_eta_losses = np.sum(region_losses, axis=1)
    avg_eta_losses = np.mean(region_losses, axis=1)

    # console.log('sum total losses')
    # console.log(sum_total_losses.shape)
    # console.log('sum_eta_losses')
    # console.log(sum_eta_losses.shape)

    sum_eta_over_total_losses = (sum_eta_losses/sum_total_losses.reshape(-1,1,1)).reshape((-1, 14))
    avg_eta_over_total_losses = (avg_eta_losses/avg_total_losses.reshape(-1,1,1)).reshape((-1, 14))

    # console.log('sum eta over sum total losses')
    # console.log(sum_eta_over_total_losses.shape)

    for eta in range(0,14):
        index_sum_eta_over_total_losses = sum_eta_over_total_losses[eta]
        index_avg_eta_over_total_losses = avg_eta_over_total_losses[eta]
        
        hep.cms.text(f'Preliminary', loc=2)
        fig = hep.histplot(
            np.histogram(
                index_sum_eta_over_total_losses,
                bins=20,
                density=True,
            )
        )
        hist_name = f'Eta{eta}_Sum_Loss_Over_Sum_Total_Loss'
        plt.xlabel(f'iEta {eta} Summed Squared Error Over Total Squared Error')
        plt.ylabel('A.U.')
        plt.yscale('log')

        plt.savefig(f'{output_path}/{hist_name}.png')
        plt.savefig(f'{output_path}/{hist_name}.pdf')
        plt.clf()

        hep.cms.text(f'Preliminary', loc=2)
        fig = hep.histplot(
            np.histogram(
                index_avg_eta_over_total_losses,
                bins=20,
                density=True,
            )
        )
        hist_name = f'Eta{eta}_Avg_Loss_Over_Avg_Total_Loss'
        plt.xlabel(f'iEta {eta} Average Squared Error Over Average Squared Error')
        plt.ylabel('A.U.')
        plt.yscale('log')

        plt.savefig(f'{output_path}/{hist_name}.png')
        plt.savefig(f'{output_path}/{hist_name}.pdf')
        plt.clf()
