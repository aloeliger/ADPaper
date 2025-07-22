#This one is going to be a touch unorthodox
#We're just going to check what fraction of the total error the leading region is responsible for in the CICADA Teacher
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

def make_leading_region_error_plot(
        sample,
        teacher_model,
        output_path
):
    #Okay, let's get the regions out of the dataframe
    #Due to the currently bugged state of regions in the Ntuples (?)
    #Let's do this from towers

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

    region_predictions = teacher_model.predict(regions_et)
    region_losses = (regions_et.reshape((-1, 18, 14, 1)) - region_predictions)**2
    total_losses = np.sum(region_losses, axis=(1,2,3))
    
    leading_region_index = np.argmax(regions_et.reshape((-1, 252)),  axis=1)

    #print(leading_region_index.shape)
    batch_indices = np.arange(regions_et.shape[0])
    leading_region_et = regions_et.reshape((-1, 252))[batch_indices, leading_region_index]
    leading_region_loss = region_losses.reshape((-1, 252))[batch_indices, leading_region_index]
    leading_region_loss_fraction_total = leading_region_loss / total_losses.reshape((-1,))

    # print(leading_region_et.shape)
    # print(leading_region_loss.shape)
    # print(total_losses.shape)
    # print(leading_region_loss_fraction_total.shape)
    
    #okay, now let's make a 2D histogram
    fig = hep.hist2dplot(
        np.histogram2d(
            leading_region_et,
            leading_region_loss_fraction_total,
            bins=20,
            range=[[0.0, 80.0], [0.0, 0.2]],
            #density=True,
        ),
        norm=colors.LogNorm()
    )

    plt.xlabel('Leading Region ET')
    plt.ylabel('Fraction of total loss')
    hist_name = f'CICADA_leading_region_fraction_total_loss'
    plt.savefig(
        f'{output_path}/{hist_name}.png'
    )
    plt.savefig(
        f'{output_path}/{hist_name}.pdf'
    )

    plt.clf()

    fig = hep.histplot(
        np.histogram(
            leading_region_et,
            bins=20,
            density=True,
            range=[0.0, 100.0],
        ),
    )
    hist_name = f'CICADA_leading_region'
    plt.xlabel('Leading Region ET')
    plt.ylabel('A.U.')

    plt.savefig(f'{output_path}/{hist_name}.png')
    plt.savefig(f'{output_path}/{hist_name}.pdf')
    plt.clf()

    hep.histplot(
        np.histogram(
            total_losses,
            bins=20,
            density=True,
            range=[0.0, 4000.0]
        )
    )
    hist_name = f'CICADA_total_loss'
    plt.xlabel('Total Teacher Loss')
    plt.ylabel('A.U.')
    plt.yscale('log')

    plt.savefig(f'{output_path}/{hist_name}.png')
    plt.savefig(f'{output_path}/{hist_name}.pdf')
    plt.clf()

    hep.histplot(
        np.histogram(
            leading_region_loss,
            bins=20,
            density=True,
            range=[0.0, 200.0]
        )
    )
    hist_name = f'CICADA_leading_region_loss'
    plt.xlabel('Leading Region Loss')
    plt.ylabel('A.U.')
    plt.yscale('log')

    plt.savefig(f'{output_path}/{hist_name}.png')
    plt.savefig(f'{output_path}/{hist_name}.pdf')
    plt.clf()

    hep.histplot(
        np.histogram(
            leading_region_loss_fraction_total,
            bins=20,
            density=True,
            range=[0.0, 0.2]
        )
    )
    hist_name = f'CICADA_fraction_total_loss'
    plt.xlabel('Leading Region Fraction of Total Loss')
    plt.ylabel('A.U.')
    plt.yscale('log')

    plt.savefig(f'{output_path}/{hist_name}.png')
    plt.savefig(f'{output_path}/{hist_name}.pdf')
    plt.clf()

    plt.close()
