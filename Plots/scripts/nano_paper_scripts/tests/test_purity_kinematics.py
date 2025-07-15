import pytest
import nano_paper_scripts.src.purity_kinematics as purity_kinematics
from .test_sample import low_stat_data_sample
import nano_paper_scripts.src.definitions as definitions
import tempfile
import os

def test_make_L1HT_purity_plot(low_stat_data_sample):
    definitions.add_all_values(low_stat_data_sample)
    definitions.add_HLT_and_scouting_values(low_stat_data_sample)
    with tempfile.TemporaryDirectory() as tempdir:
        purity_kinematics.make_L1HT_purity_plot(
            sample = low_stat_data_sample,
            sample_name = 'dummy',
            score_name = 'CICADA2024_CICADAScore',
            score_display_name = 'CICADA 2024',
            score_label='CICADA Dummy',
            score_value=5.0,
            output_dir=tempdir,
        )
