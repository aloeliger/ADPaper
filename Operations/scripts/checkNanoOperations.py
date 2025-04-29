import ROOT
from rich.console import Console
from rich.table import Table
from rich.progress import track
import numpy as np

console = Console()

def main():
    console.log('Start')
    file_name = 'operations_file.root'
    the_chain = ROOT.TChain('Events')
    the_chain.Add(file_name)
    #df = ROOT.RDataFrame(the_chain)

    #the_chain.Print()

    nEvents = the_chain.GetEntries()
    console.log(f'Events: {nEvents}')

    average_input_discrepancy = np.zeros((18,14))
    matching_inputs = 0

    info_table = Table(title='Event Logs')
    info_table.add_column('Emu CICADA 2024 (Sim Inputs)')
    info_table.add_column('Emu CICADA 2024 (Unpacked Inputs)')
    info_table.add_column('Emu CICADA 2025 (Sim Inputs)')
    info_table.add_column('Emu CICADA 2025 (Unpacked Inputs)')
    info_table.add_column('Unpacked CICADA')
    info_table.add_column('Sim Inputs Match Unpacked?')
    
    for i in track(range(nEvents)):
        the_chain.GetEntry(i)

        sim_inputs = np.zeros((18,14))
        unpacked_inputs = np.zeros((18,14))
        for j in range(the_chain.nRegions):
            unpacked_inputs[the_chain.Regions_iphi[j]][the_chain.Regions_ieta[j]] = the_chain.Regions_et[j]
        for j in range(the_chain.nSimRegions):
            unpacked_inputs[the_chain.SimRegions_iphi[j]][the_chain.SimRegions_ieta[j]] = the_chain.SimRegions_et[j]
        same_inputs = np.array_equal(sim_inputs, unpacked_inputs)
        if same_inputs:
            matching_inputs += 1
        else:
            average_input_discrepancy += (sim_inputs - unpacked_inputs)
        #console.log(same_inputs)
        cicada_2024_score = the_chain.CICADA2024_CICADAScore
        cicada_2024_unpacked_score = the_chain.CICADA2024_Unpacked_CICADAScore
        cicada_2025_score = the_chain.CICADA2025_CICADAScore
        cicada_2025_unpacked_score = the_chain.CICADA2025_Unpacked_CICADAScore
        cicada_unpacked_score = the_chain.CICADAUnpacked_CICADAScore

        if sum([cicada_2024_score, cicada_2024_unpacked_score, cicada_2025_score, cicada_2025_unpacked_score, cicada_unpacked_score]) != 0.0:
            info_table.add_row(
                f'{cicada_2024_score}',
                f'{cicada_2024_unpacked_score}',
                f'{cicada_2025_score}',
                f'{cicada_2025_unpacked_score}',
                f'{cicada_unpacked_score}',
                f'{same_inputs}'
            )
    try:
        average_input_discrepancy = average_input_discrepancy / (nEvents-matching_inputs)
    except ZeroDivisionError:
        pass

    console.log('Event log (at least one non zero score):')
    console.print(info_table)
    console.log(f'Matching inputs: {matching_inputs}/{nEvents} ~= {matching_inputs/nEvents:.2%}')
    console.log('Average input discrepancy:')
    console.print(average_input_discrepancy)

    console.log('[green]Done![/green]')

if __name__ == '__main__':
    main()
