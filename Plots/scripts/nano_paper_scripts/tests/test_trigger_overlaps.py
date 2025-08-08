import pytest
import nano_paper_scripts.src.trigger_overlaps as trigger_overlaps
import nano_paper_scripts.src.definitions as definitions
from .test_sample import low_stat_data_sample
import tempfile
import os

def test_make_total_trigger_overlap_plot(low_stat_data_sample):
    definitions.add_all_values(low_stat_data_sample)
    with tempfile.TemporaryDirectory() as tempdir:
        trigger_overlaps.make_total_trigger_overlap_plot(
            sample=low_stat_data_sample,
            sample_name='dummy',
            output_path = tempdir
        )
        assert(os.listdir(tempdir))

def test_sample_leading_trigger_overlap_plot(low_stat_data_sample):
    definitions.add_all_values(low_stat_data_sample)
    with tempfile.TemporaryDirectory() as tempdir:
        trigger_overlaps.make_total_trigger_overlap_plot(
            sample=low_stat_data_sample,
            sample_name='dummy',
            output_path = tempdir
        )

        assert(os.listdir(tempdir))
