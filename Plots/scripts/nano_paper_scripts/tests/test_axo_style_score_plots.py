import pytest
import nano_paper_scripts.src.definitions as definitions
import nano_paper_scripts.src.axo_style_score_plots as axo_style_score_plots
import tempfile
import os

from .test_sample import low_stat_data_sample

def test_make_axo_style_score_plot(low_stat_data_sample):
    definitions.add_all_values(low_stat_data_sample)
    with tempfile.TemporaryDirectory() as tempdir:
        axo_style_score_plots.make_axo_style_score_plot(
            low_stat_data_sample,
            score = 'CICADA2024_CICADAScore',
            score_name = 'dummy',
            working_point = 5.0,
            output_path = tempdir
        )

        assert(os.listdir(tempdir))
