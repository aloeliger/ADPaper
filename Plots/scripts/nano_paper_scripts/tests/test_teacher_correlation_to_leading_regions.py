import pytest
import nano_paper_scripts.src.teacher_correlation_to_leading_regions as teacher_correlation_to_leading_regions
from .test_sample import low_stat_data_sample
import tempfile
import os
import huggingface_hub

def test_make_leading_region_error_plot(low_stat_data_sample):
    model = huggingface_hub.from_pretrained_keras('cicada-project/teacher-v.0.1')
    with tempfile.TemporaryDirectory() as tempdir:
        teacher_correlation_to_leading_regions.make_leading_region_error_plot(
            sample=low_stat_data_sample,
            teacher_model = model,
            output_path = tempdir
        )

        assert(os.listdir(tempdir))
