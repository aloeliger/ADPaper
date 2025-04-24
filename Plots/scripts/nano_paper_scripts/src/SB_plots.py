import mplhep as hep
import matplotlib.pyplot as plt
import numpy as np
import ROOT
from pathlib import Path
from rich.console import Console

from .sample import construct_data_samples
from .config import Configuration

console = Console()

def make_S_over_SPlusB_Plot(
        background_sample,
        signal_sample,
        anomalyScoreName,
        anomalyScore,
        output_path,
):
    pass
