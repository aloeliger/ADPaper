import pytest
import nano_paper_scripts.src.pure_roc_plots as pure_roc_plots
import nano_paper_scripts.src.definitions as definitions
from .test_sample import low_stat_data_sample, low_stat_signal_sample
import tempfile
import os

def test_make_pure_roc_plot(low_stat_data_sample, low_stat_signal_sample):
    definitions.add_all_values(low_stat_data_sample)
    definitions.add_all_values(low_stat_signal_sample)
    with tempfile.TemporaryDirectory() as tempdir:
        pure_roc_plots.make_pure_roc_plot(
            {'dummy': low_stat_signal_sample},
            ['dummy'],
            low_stat_data_sample,
            'dummy',
            'CICADA2024',
            'CICADA2024_CICADAScore',
            tempdir
        )

        assert(os.listdir(tempdir))
