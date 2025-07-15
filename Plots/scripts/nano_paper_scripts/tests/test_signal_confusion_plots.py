import pytest
import nano_paper_scripts.src.signal_confusion_plots as signal_confusion_plots
from .test_sample import low_stat_signal_sample, low_stat_data_sample
import tempfile
import os
import nano_paper_scripts.src.definitions as definitions

def test_make_confusion_plot(low_stat_signal_sample):
    definitions.add_all_values(low_stat_signal_sample)
    with tempfile.TemporaryDirectory() as tempdir:
        signal_confusion_plots.make_confusion_plot(
            sample=low_stat_signal_sample,
            sample_name = 'dummy_signal',
            score_display_name = 'CICADA 2024',
            score_name='CICADA2024_CICADAScore',
            score_value=5.0,
            output_dir=tempdir
        )

        assert(os.listdir(tempdir))
