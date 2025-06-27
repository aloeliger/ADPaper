import src.overlap_table as overlap_table
from src.config import Configuration
from src.definitions import add_all_values
from src.sample import construct_data_samples
from pathlib import Path

from rich.console import Console

console = Console()

def main():
    console.log('Making overlap tables and overlap table plots')

    current_config = Configuration.GetConfiguration().configs

    output_path = Path(current_config['output path']+'/Overlap_table_plots/')

    output_path.mkdir(exist_ok=True, parents=True)

    data_sample = construct_data_samples()['RunI']
    add_all_values(data_sample)

    cicada_names = current_config['CICADA Scores']
    axo_names = current_config['AXO Scores']

    for cicada_name in cicada_names:
        console.log(f'CICADA Name: {cicada_name}')
        working_points = current_config['CICADA working points'][cicada_name]
        for working_point in working_points:
            working_point_name = working_point.replace(' ', '')
            the_overlap_table = overlap_table.build_emulated_overlap_table(
                data_sample,
                score = cicada_names[cicada_name],
                score_name = cicada_name,
                working_point = working_points[working_point],
                list_of_triggers = current_config['unprescaled triggers']
            )

            overlap_table.make_emulated_max_overlap_table_plot(
                the_overlap_table,
                output_path=output_path,
                hist_name = f'overlap_table_{cicada_name}_{working_point_name}',
                n_triggers_to_use=10
            )
            overlap_table.print_overlap_table(the_overlap_table, n_triggers=10)
            
    for axo_name in axo_names:
        console.log(f'AXO Name: {cicada_name}')
        working_points = current_config['AXO working points'][axo_name]
        for working_point in working_points:
            working_point_name = working_point.replace(' ', '')
            the_overlap_table = overlap_table.build_emulated_overlap_table(
                data_sample,
                score = axo_names[axo_name],
                score_name = axo_name,
                working_point = working_points[working_point],
                list_of_triggers = current_config['unprescaled triggers']
            )

            overlap_table.make_emulated_max_overlap_table_plot(
                the_overlap_table,
                output_path=output_path,
                hist_name = f'overlap_table_{axo_name}_{working_point_name}',
                n_triggers_to_use=10
            )
            overlap_table.print_overlap_table(the_overlap_table, n_triggers=10)

if __name__ =='__main__':
    main()
