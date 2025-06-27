import pytest
import nano_paper_scripts.src.definitions as definitions
import nano_paper_scripts.src.overlap_table as overlap_table
import tempfile
import os

from .test_sample import low_stat_data_sample

@pytest.fixture
def unprescaled_trigger_list():
    return [
        "L1_SingleJet180",
	"L1_SingleJet180er2p5",
	"L1_SingleJet200",
	"L1_SingleJet43er2p5_NotBptxOR_3BX",
	"L1_SingleJet46er2p5_NotBptxOR_3BX",
	"L1_SingleMu22",
	"L1_SingleMu25",
    ]

def test_build_emulated_overlap_table(low_stat_data_sample, unprescaled_trigger_list):
    definitions.add_all_values(low_stat_data_sample)
    the_overlap_table = overlap_table.build_emulated_overlap_table(
        sample=low_stat_data_sample,
        score = 'CICADA2024_CICADAScore',
        score_name='dummy',
        working_point=5.0,
        list_of_triggers = unprescaled_trigger_list 
    )

    assert(isinstance(the_overlap_table, list))

def test_make_emulated_max_overlap_table_plot(low_stat_data_sample, unprescaled_trigger_list):
    definitions.add_all_values(low_stat_data_sample)
    the_overlap_table = overlap_table.build_emulated_overlap_table(
        sample=low_stat_data_sample,
        score = 'CICADA2024_CICADAScore',
        score_name='dummy',
        working_point=5.0,
        list_of_triggers = unprescaled_trigger_list 
    )

    with tempfile.TemporaryDirectory() as tempdir:
        overlap_table.make_emulated_max_overlap_table_plot(
            the_overlap_table,
            output_path = tempdir,
            hist_name = 'dummy',
            n_triggers_to_use=5
        )

        assert(os.listdir(tempdir))

def test_print_overlap_table(low_stat_data_sample, unprescaled_trigger_list):
    definitions.add_all_values(low_stat_data_sample)
    the_overlap_table = overlap_table.build_emulated_overlap_table(
        sample=low_stat_data_sample,
        score = 'CICADA2024_CICADAScore',
        score_name='dummy',
        working_point=5.0,
        list_of_triggers = unprescaled_trigger_list 
    )

    overlap_table.print_overlap_table(the_overlap_table)
