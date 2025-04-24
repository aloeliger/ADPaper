from src.axo_style_score_plots import make_axo_style_score_plot
from src.config import Configuration
from src.definitions import add_all_values
from src.sample import construct_data_samples
from pathlib import Path

from rich.console import Console

console = Console()

def main():
    console.log('Making AXO style score plots')

    current_config = Configuration.GetConfiguration().configs

    output_path = Path(current_config['output path']+'/AXO_style_score_plots/')
    output_path.mkdir(exist_ok=True, parents=True)

    data_sample = construct_data_samples()['RunI']
    add_all_values(data_sample)

    cicada_names = current_config['CICADA Scores']

    for cicada_name in cicada_names:
        make_axo_style_score_plot(
            sample = data_sample,
            score = current_config['CICADA Scores'][cicada_name],
            score_name = cicada_name,
            working_point = current_config['CICADA working points'][cicada_name]['CICADA Medium'],
            output_path = output_path
        )
    
    console.log('[green]Done making AXO style score plots![/green]')

if __name__ == '__main__':
    main()
