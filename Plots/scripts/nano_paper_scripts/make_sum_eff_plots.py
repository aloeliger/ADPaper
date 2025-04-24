from src.eff_plots import make_var_eff_plot
from src.config import Configuration
from src.definitions import add_all_values
from src.sample import construct_data_samples
from pathlib import Path

from rich.console import Console

console = Console()

def main():
    console.log('Making Sum Eff Plots')
    current_config = Configuration.GetConfiguration().configs

    output_path = Path(current_config['output path']+'/sum_eff_plots/')
    output_path.mkdir(exist_ok=True, parents=True)

    data_sample = construct_data_samples()['RunI']
    add_all_values(data_sample)

    cicada_names = current_config['CICADA Scores']

    sums = {
        'L1_HT': ('L1JetHT', 'L1 Jet $H_{T}$ [GeV]', 13, 0.0, 1300.0),
        'L1_MET': ('L1MET', 'L1 $p_{T}^{miss}$ [GeV]', 25, 0.0, 250.0),
        'L1EG_pt_sum': ('EGPTSum', 'L1EG $\Sigma p_{T}$ [GeV]', 10, 0.0, 1000.0)
    }

    for cicada_name in cicada_names:
        for variable in sums:
            make_var_eff_plot(
                data_sample,
                cicada_name,
                cicada_score = current_config['CICADA Scores'][cicada_name],
                working_point = current_config['CICADA working points'][cicada_name]['CICADA Medium'],
                var_name = variable,
                hist_tag = sums[variable][0],
                xaxis_label = sums[variable][1],
                nbins = sums[variable][2],
                xaxis_min=sums[variable][3],
                xaxis_max=sums[variable][4],
                output_path = output_path
            )

    console.log('[green]Making Sum Eff Plots Done![/green]')

if __name__ == '__main__':
    main()
