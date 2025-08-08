import pytest
import nano_paper_scripts.src.cluster_comparison as cluster_comparison
import nano_paper_scripts.src.definitions as definitions
from .test_sample import low_stat_data_sample, low_stat_signal_sample
import tempfile
import os

def test_make_cluster_comparison_plot(low_stat_data_sample, low_stat_signal_sample):
    definitions.add_all_values(low_stat_data_sample)
    definitions.add_all_values(low_stat_signal_sample)

    with tempfile.TemporaryDirectory() as tempdir:
        cluster_comparison.make_cluster_comparison_plot(
            background_sample = low_stat_data_sample,
            background_sample_name = 'Zero Bias',
            samples = {
                'TT': low_stat_signal_sample
            },
            score_name = 'CICADA2024_CICADAScore',
            score_display_name='CICADA2024',
            output_path=tempdir,
        )

def test_make_tower_variable(low_stat_data_sample):
    definitions.add_all_values(low_stat_data_sample)

    low_stat_data_sample = cluster_comparison.make_tower_variable(low_stat_data_sample)

    all_columns = low_stat_data_sample.df.GetColumnNames()
    assert('emul_calo_tower_bx_0_mask' in all_columns)
    assert('nL1EmulCaloTower_BX0' in all_columns)
    
        
def test_make_tower_comparison_plot(low_stat_data_sample, low_stat_signal_sample):
    definitions.add_all_values(low_stat_data_sample)
    definitions.add_all_values(low_stat_signal_sample)
    cluster_comparison.make_tower_variable(low_stat_data_sample)
    cluster_comparison.make_tower_variable(low_stat_signal_sample)

    with tempfile.TemporaryDirectory() as tempdir:
        cluster_comparison.make_tower_comparison_plot(
            background_sample = low_stat_data_sample,
            background_sample_name = 'Zero Bias',
            samples = {
                'TT': low_stat_signal_sample
            },
            score_name = 'CICADA2024_CICADAScore',
            score_display_name='CICADA2024',
            output_path=tempdir,
        )
