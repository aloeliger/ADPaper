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
