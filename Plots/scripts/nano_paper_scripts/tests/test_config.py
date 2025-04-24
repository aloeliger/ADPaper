import pytest
import nano_paper_scripts.src.config as config

def test_Configuration():
    the_config = config.Configuration.GetConfiguration()

    assert(the_config is not None)
    assert(isinstance(the_config, config.Configuration))
