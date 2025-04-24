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

def make_var_eff_plot(
        sample,
        cicada_name,
        cicada_score,
        working_point,
        var_name,
        nbins,
        hist_tag,
        xaxis_label,
        xaxis_min,
        xaxis_max,
        output_path,
        yaxis_min = None,
        yaxis_max = None,
):
    hep.style.use("CMS")
    hep.cms.text("Preliminary", loc=2)

    values = sample.df.AsNumpy(
        [
            cicada_score,
            var_name,
        ]
    )
    cicada_scores = values[cicada_score]
    objects = values[var_name]

    WP_mask = (cicada_scores > working_point)
    objects_WorkingPoint = objects[WP_mask]

    overall_hist = np.histogram(
        objects,
        bins=nbins,
        range=(xaxis_min, xaxis_max)
    )
    overall_fig = hep.histplot(
        overall_hist,
        label='All Zero Bias'
    )

    wp_hist = np.histogram(
        objects_WorkingPoint,
        bins=nbins,
        range=(xaxis_min, xaxis_max)
    )
    wp_fig = hep.histplot(
        wp_hist,
        label='CICADA Nominal',
    )

    plt.legend(loc='upper right', title='Zero Bias Triggered Events')
    plt.xlabel(f'{xaxis_label}')
    plt.ylabel('Events')
    plt.yscale('log')
    plt.ylim(1.0, np.max(overall_hist[0]*100.0))
    plt.grid(c='lightgray')

    hist_name = f'{cicada_name}_{hist_tag}'

    plt.savefig(
        f'{output_path}/{hist_name}.png'
    )
    plt.savefig(
        f'{output_path}/{hist_name}.pdf'
    )
    plt.close()

    hep.style.use("CMS")
    hep.cms.text("Preliminary", loc=2)

    eff_plot = (wp_hist[0]/overall_hist[0]) * 100.0
    eff_errors = (np.sqrt(wp_hist[0]) / overall_hist[0]) * 100.0
    #eff_errors_up = np.clip(eff_plot + eff_errors, 0.0, 100.0)
    #eff_errors_down = np.clip(eff_plot - eff_errors, 0.0, 100.0)
    #eff_errors = np.array([eff_errors_down, eff_errors_up])
    eff_errors_up = np.copy(eff_errors)
    eff_errors_down = np.copy(eff_errors)
    for i in range(len(eff_errors)):
        if eff_plot[i] + eff_errors[i] > 100.0:
            eff_errors_up[i] = 100.0-eff_plot[i]
        if eff_plot[i] - eff_errors[i] < 0.0:
            eff_errors_down = eff_plot[i]
    
    plt.errorbar(
        overall_hist[1][:-1],
        eff_plot,
        xerr = np.zeros(len(eff_plot)),
        yerr = eff_errors,
        c = 'orange',
        label='CICADA Nominal'
    )

    plt.legend(loc='upper right', title='Zero Bias Triggered Events')
    plt.xlabel(f'{xaxis_label}')
    plt.ylabel('Acceptance [%]')
    plt.yscale('linear')

    used_min = 0.0
    used_max = 140.0
    if yaxis_min is not None:
        used_min = yaxis_min
    if yaxis_max is not None:
        used_max = yaxis_max
    plt.ylim(used_min, used_max)
    plt.axhline(y=100.0, linestyle='--', c='grey')
    plt.grid(c='lightgray')

    hist_name = f'{cicada_name}_{hist_tag}_eff'
    plt.savefig(
        f'{output_path}/{hist_name}.png'
    )
    plt.savefig(
        f'{output_path}/{hist_name}.pdf'
    )
    plt.close()
    
def main():
    console.log('Making Eff and Object plots')
    current_config = Configuration.GetConfiguration().configs

    output_path = Path(current_config['output path']+'/eff_plots/')
    output_path.mkdir(exist_ok=True, parents=True)

    data_sample = construct_data_samples()['RunI']
    add_all_values(data_sample)

    cicada_names = current_config['CICADA Scores']

    for cicada_name in cicada_names:
        #
        # HT Plot
        #
        make_var_eff_plot(
            data_sample,
            cicada_name,
            cicada_score = current_config['CICADA Scores'][cicada_name],
            working_point = current_config['CICADA working points'][cicada_name]['CICADA Medium'],
            sum_name = 'L1_HT',
            nbins=13,
            hist_tag = 'L1JetHT',
            xaxis_label = 'L1 Jet $H_{T}$ [GeV]',
            xaxis_min=0.0,
            xaxis_max=1300.0,
            output_path = output_path
        )

        make_var_eff_plot(
            data_sample,
            cicada_name,
            cicada_score = current_config['CICADA Scores'][cicada_name],
            working_point = current_config['CICADA working points'][cicada_name]['CICADA Medium'],
            sum_name = 'L1_MET',
            nbins=25,
            hist_tag = 'L1MET',
            xaxis_label = 'L1 $p_{T}^{miss}$ [GeV]',
            xaxis_min = 0.0,
            xaxis_max = 250.0,
            output_path = output_path
        )

        make_var_eff_plot(
            data_sample,
            cicada_name,
            cicada_score = current_config['CICADA Scores'][cicada_name],
            working_point = current_config['CICADA working points'][cicada_name]['CICADA Medium'],
            sum_name = 'L1EG_pt_sum',
            nbins=10,
            hist_tag = 'EGPTSum',
            xaxis_label = 'L1EG $\Sigma p_{T}$ [GeV]',
            xaxis_min = 0.0,
            xaxis_max = 1000.0,
            output_path = output_path
        )

        #
        # Object plots
        #

        make_var_eff_plot(
            data_sample,
            cicada_name,
            cicada_score = current_config['CICADA Scores'][cicada_name],
            working_point = current_config['CICADA working points'][cicada_name]['CICADA Medium'],
            nbins=14,
            var_name='nJet',
            hist_tag = 'nJet',
            xaxis_label = '$N_{Jet}$',
            xaxis_min = 0.0,
            xaxis_max = 14.0,
            output_path = output_path,
        )
        make_var_eff_plot(
            data_sample,
            cicada_name,
            cicada_score = current_config['CICADA Scores'][cicada_name],
            working_point = current_config['CICADA working points'][cicada_name]['CICADA Medium'],
            nbins=14,
            var_name = 'nMuon',
            hist_tag = 'nMuon',
            xaxis_label = '$N_{\mu}$',
            xaxis_min = 0.0,
            xaxis_max = 14.0,
            output_path = output_path,
        )
        make_var_eff_plot(
            data_sample,
            cicada_name,
            cicada_score = current_config['CICADA Scores'][cicada_name],
            working_point = current_config['CICADA working points'][cicada_name]['CICADA Medium'],
            nbins=14,
            var_name='nElectron',
            hist_tag = 'nElectron',
            xaxis_label = '$N_{Electron}$',
            xaxis_min = 0.0,
            xaxis_max = 14.0,
            output_path = output_path,
        )
        make_var_eff_plot(
            data_sample,
            cicada_name,
            cicada_score = current_config['CICADA Scores'][cicada_name],
            working_point = current_config['CICADA working points'][cicada_name]['CICADA Medium'],
            nbins=14,
            var_name='nTau',
            hist_tag = 'nTau',
            xaxis_label = '$N_{\\tau}$',
            xaxis_min = 0.0,
            xaxis_max = 14.0,
            output_path = output_path,
        )

        #
        # L1 Object Plots
        #
        make_var_eff_plot(
            data_sample,
            cicada_name,
            cicada_score = current_config['CICADA Scores'][cicada_name],
            working_point = current_config['CICADA working points'][cicada_name]['CICADA Medium'],
            nbins=14,
            var_name = 'nL1EmulJet',
            hist_tag = 'nL1EmulJet',
            xaxis_label = '$N_{L1 Jet}$',
            xaxis_min = 0.0,
            xaxis_max = 14.0,
            output_path = output_path,
        )

        make_var_eff_plot(
            data_sample,
            cicada_name,
            cicada_score = current_config['CICADA Scores'][cicada_name],
            working_point = current_config['CICADA working points'][cicada_name]['CICADA Medium'],
            nbins=14,
            var_name = 'nL1EmulMu',
            hist_tag = 'nL1EmulMu',
            xaxis_label = '$N_{L1 \mu}$',
            xaxis_min = 0.0,
            xaxis_max = 14.0,
            output_path = output_path,
        )

        make_var_eff_plot(
            data_sample,
            cicada_name,
            cicada_score = current_config['CICADA Scores'][cicada_name],
            working_point = current_config['CICADA working points'][cicada_name]['CICADA Medium'],
            nbins=12,
            var_name = 'nL1EmulEG',
            hist_tag = 'nL1EmulEG',
            xaxis_label = '$N_{L1 EG}$',
            xaxis_min = 0.0,
            xaxis_max = 12.0,
            output_path = output_path,
        )

        make_var_eff_plot(
            data_sample,
            cicada_name,
            cicada_score = current_config['CICADA Scores'][cicada_name],
            working_point = current_config['CICADA working points'][cicada_name]['CICADA Medium'],
            nbins=12,
            var_name = 'nL1EmulTau',
            hist_tag = 'nL1EmulTau',
            xaxis_label = '$N_{L1 EG}$',
            xaxis_min = 0.0,
            xaxis_max = 12.0,
            output_path = output_path,
        )
    console.log('[green]Done![/green]')
