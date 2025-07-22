import pytest
import nano_paper_scripts.src.leading_object_scatters as leading_object_scatters
from .test_sample import low_stat_data_sample
import tempfile
import os

def test_make_leading_object_scatter(low_stat_data_sample):
    with tempfile.TemporaryDirectory() as tempdir:
        leading_object_scatters.make_leading_object_scatter(
            sample = low_stat_data_sample,
            sample_name = 'dummy_data',
            score_name = 'CICADA2024_CICADAScore',
            score_display_name = 'CICADA2024',
            object_to_use = 'Jet',
            output_path = tempdir
        )
