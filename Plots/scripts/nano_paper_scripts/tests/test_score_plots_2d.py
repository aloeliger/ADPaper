import pytest
import nano_paper_scripts.src.score_plots_2D as score_plots_2D
from .test_sample import low_stat_data_sample
import tempfile
import os

def test_make_scatter_plot(low_stat_data_sample):
    with tempfile.TemporaryDirectory() as tempdir:
        score_plots_2D.make_scatter_plot(
            low_stat_data_sample,
            'dummy_cicada',
            'dummy_axo',
            'CICADA2024_CICADAScore',
            'axol1tl_v4_AXOScore',
            tempdir
        )

        assert(os.listdir(tempdir))
