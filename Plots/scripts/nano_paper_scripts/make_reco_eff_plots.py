from src.eff_plots import make_var_eff_plot
from src.config import Configuration
from src.definitions import add_all_values
from src.sample import construct_data_samples
from pathlib import Path

from rich.console import Console

console = Console()

def main():
    console.log('Making Reco Eff Plots')
    current_config = Configuration.GetConfiguration().configs

    output_path = Path(current_config['output path']+'/reco_object_eff_plots/')
    output_path.mkdir(exist_ok=True, parents=True)

    data_sample = construct_data_samples()['RunI']
    add_all_values(data_sample)

    cicada_names = current_config['CICADA Scores']

    variables = {
        'nJet': ('nJet', '$N_{Jet}$', 14, 0.0, 14.0),
        'nMuon': ('nMuon', '$N_{\mu}', 14, 0.0, 14.0),
        'nElectron': ('nElectron', '$N_{Electron}$', 14, 0.0, 14.0),
        'nTau': ('nTau', '$N_{\\tau}$', 14, 0.0, 14.0),
    }

    for cicada_name in cicada_names:
        for variable in variables:
            make_var_eff_plot(
                data_sample,
                cicada_name,
                cicada_score = current_config['CICADA Scores'][cicada_name],
                working_point = current_config['CICADA working points'][cicada_name]['CICADA Medium'],
                var_name = variable,
                hist_tag = variables[variable][0],
                xaxis_label = variables[variable][1],
                nbins = variables[variable][2],
                xaxis_min=variables[variable][3],
                xaxis_max=variables[variable][4],
                output_path = output_path
            )

    console.log('[green]Reco Eff Plots Done![/green]')

if __name__ == '__main__':
    main()
