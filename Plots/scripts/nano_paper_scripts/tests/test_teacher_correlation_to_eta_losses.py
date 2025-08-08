import pytest
import nano_paper_scripts.src.teacher_correlation_to_eta_losses as teacher_correlation_to_eta_losses
from .test_sample import low_stat_data_sample
import tempfile
import os
import huggingface_hub

def test_make_eta_losses_plot(low_stat_data_sample):
    model = huggingface_hub.from_pretrained_keras('cicada-project/teacher-v.0.1')
    with tempfile.TemporaryDirectory() as tempdir:
        teacher_correlation_to_eta_losses.make_eta_losses_plot(
            sample=low_stat_data_sample,
            teacher_model = model,
            output_path = tempdir
        )

        assert(os.listdir(tempdir))
