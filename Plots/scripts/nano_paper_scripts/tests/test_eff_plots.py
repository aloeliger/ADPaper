import pytest
import nano_paper_scripts.src.eff_plots as eff_plots
from .test_sample import low_stat_data_sample
import tempfile
import os

def test_make_var_eff_plot(low_stat_data_sample):
    with tempfile.TemporaryDirectory() as tempdir:
        eff_plots.make_var_eff_plot(
            low_stat_data_sample,
            'dummy_cicada',
            'CICADA2024_CICADAScore',
            121.0,
            var_name = 'nL1EmulJet',
            nbins=10,
            hist_tag = 'nL1EmulJet',
            xaxis_label='dummy cicada',
            xaxis_min = 0.0,
            xaxis_max = 10.0,
            output_path = tempdir,
        )

        assert(os.listdir(tempdir))
