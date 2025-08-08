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
        values,
        #sample,
        teacher_model,
        output_path
):
    hep.style.use("CMS")
    hep.cms.text(f'Preliminary', loc=2)
    #Okay, let's get the regions out of the dataframe
    #Due to the currently bugged state of regions in the Ntuples (?)
    #Let's do this from towers

    #values = sample.df.AsNumpy(['Regions_et','Regions_ieta', 'Regions_iphi'])

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
    sum_total_losses = np.sum(region_losses, axis=(1,2,3))
    avg_total_losses = np.mean(region_losses, axis=(1,2,3))
    
    leading_region_index = np.argmax(regions_et.reshape((-1, 252)),  axis=1)

    #print(leading_region_index.shape)
    batch_indices = np.arange(regions_et.shape[0])
    leading_region_et = regions_et.reshape((-1, 252))[batch_indices, leading_region_index]
    leading_region_loss = region_losses.reshape((-1, 252))[batch_indices, leading_region_index]
    leading_region_loss_fraction_total = leading_region_loss / sum_total_losses.reshape((-1,))
    leading_region_loss_fraction_avg = leading_region_loss / avg_total_losses.reshape((-1,))

    losses_without_leading_region = np.copy(region_losses.reshape((-1, 18*14)))
    losses_without_leading_region[leading_region_index] = 0.0
    mse_without_leading_region = np.sum(losses_without_leading_region, axis=1)/(251.0)
    mse_with_leading_region = np.mean(region_losses.reshape((-1, 18*14)), axis=1)

    # print(leading_region_et.shape)
    # print(leading_region_loss.shape)
    # print(total_losses.shape)
    # print(leading_region_loss_fraction_total.shape)
    
    #okay, now let's make a 2D histogram
    hep.cms.text(f'Preliminary', loc=2)
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
    plt.ylabel('Fraction of summed region losses')
    hist_name = f'CICADA_leading_region_fraction_total_loss'
    plt.savefig(
        f'{output_path}/{hist_name}.png'
    )
    plt.savefig(
        f'{output_path}/{hist_name}.pdf'
    )

    plt.clf()

    hep.cms.text(f'Preliminary', loc=2)
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
            sum_total_losses,
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

    hep.cms.text(f'Preliminary', loc=2)
    hep.histplot(
        np.histogram(
            avg_total_losses,
            bins=20,
            density=True,
        )
    )
    hist_name = f'CICADA_total_loss'
    plt.xlabel('Total Average Loss (MSE)')
    plt.ylabel('A.U.')
    plt.yscale('log')

    plt.savefig(f'{output_path}/{hist_name}.png')
    plt.savefig(f'{output_path}/{hist_name}.pdf')
    plt.clf()

    hep.cms.text(f'Preliminary', loc=2)
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

    hep.cms.text(f'Preliminary', loc=2)
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

    hep.cms.text(f'Preliminary', loc=2)
    hep.histplot(
        np.histogram(
            leading_region_loss_fraction_avg,
            bins=20,
            density=True,
        )
    )
    hist_name = f'CICADA_fraction_avg_loss'
    plt.xlabel('Leading Region Fraction of Avg Loss')
    plt.ylabel('A.U.')
    plt.yscale('log')

    plt.savefig(f'{output_path}/{hist_name}.png')
    plt.savefig(f'{output_path}/{hist_name}.pdf')
    plt.clf()

    hep.cms.text(f'Preliminary', loc=2)
    hep.histplot(
        np.histogram(
            mse_with_leading_region,
            bins=20,
            density=True,
        ),
        label='Teacher M.S.E Including Leading Region'
    )
    hep.histplot(
        np.histogram(
            mse_without_leading_region,
            bins=20,
            density=True,
        ),
        label='Teacher M.S.E Without Leading Region'
    )
    hist_name = f'CICADA_Loss_Comparisons'
    plt.xlabel('Teacher Loss')
    plt.legend()
    plt.ylabel('A.U.')
    plt.yscale('log')

    plt.savefig(f'{output_path}/{hist_name}.png')
    plt.savefig(f'{output_path}/{hist_name}.pdf')
    plt.clf()

    hep.cms.text(f'Preliminary', loc=2)
    hep.histplot(
        np.histogram(
            mse_with_leading_region - mse_without_leading_region,
            bins=20,
            density=True,
        ),
    )

    hist_name = f'CICADA_Loss_Comparisons'
    plt.xlabel('Difference between loss with and without leading region')
    plt.legend()
    plt.ylabel('A.U.')
    plt.yscale('log')

    plt.savefig(f'{output_path}/{hist_name}.png')
    plt.savefig(f'{output_path}/{hist_name}.pdf')
    plt.clf()
    
    plt.close()
    
def make_eta_losses_plot(
        values,
        #sample,
        teacher_model,
        output_path
):
    hep.style.use("CMS")
    hep.cms.text(f'Preliminary', loc=2)

    #values = sample.df.AsNumpy(['Regions_et','Regions_ieta', 'Regions_iphi'])

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
