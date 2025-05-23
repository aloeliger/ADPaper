import pytest
import nano_paper_scripts.src.utilities as utilities
import numpy as np

def test_convert_eff_to_rate():
    effs = np.array([0.0, 0.5, 1.0])
    utilities.convert_eff_to_rate(effs)

def test_convert_eff_to_rate_2():
    eff = 0.0
    rate = utilities.convert_eff_to_rate(eff)
    assert(rate==0.0)

def test_convert_rate_to_eff():
    rates = np.array([0.1, 1.0])
    utilities.convert_rate_to_eff(rates)
