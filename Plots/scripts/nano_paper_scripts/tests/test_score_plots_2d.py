import pytest
import nano_paper_scripts.src.score_plots_2D as score_plots_2D
from .test_sample import low_stat_data_sample
import tempfile
import os

def test_make_scatter_plot(low_stat_data_sample):
    with tempfile.TemporaryDirectory() as tempdir:
        score_plots_2D.make_scatter_plot(
            sample = low_stat_data_sample,
            sample_name = 'dummy_data',
            cicada_name = 'dummy_cicada',
            axo_name = 'dummy_axo',
            cicada_score = 'CICADA2024_CICADAScore',
            axo_score = 'axol1tl_v4_AXOScore',
            output_path = tempdir
        )

        assert(os.listdir(tempdir))
