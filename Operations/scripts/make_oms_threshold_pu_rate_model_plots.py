import h5py
import numpy as np
import argparse
import pickle
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.gaussian_process import GaussianProcessRegressor
import sklearn.gaussian_process as gp
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from rich.console import Console

import matplotlib.pyplot as plt

console = Console()

cicada_thresholds = {
    'L1_CICADA_VLoose': 50.0,
    'L1_CICADA_Loose': 60.0,
    'L1_CICADA_Medium': 70.0,
    'L1_CICADA_Tight': 80.0,
    'L1_CICADA_VTight': 90.0,
    'L1_CICADA_VVTight': 110.0,
    'L1_CICADA_VVVTight': 125.0,
    'L1_CICADA_VVVVTight': 135.0,
}

def make_pu_model_plot(
        model,
        threshold,
        name,
        title,
        output_path,
        train_threshold_pu_rate_examples=None,
        val_threshold_pu_rate_examples=None):
    pu_test = np.linspace(0.0, 68.0, num=100)
    x_test = np.append(
        np.tile(threshold, reps=100).reshape((-1,1)),
        pu_test.reshape((-1, 1)),
        axis=1
    )
    model_mean, model_std =model.predict(x_test, return_std=True)

    plt.plot(
        pu_test,
        model_mean,
        label='Model',
        c='orange',
    )
    plt.fill_between(
        pu_test,
        model_mean - 1.96*model_std,
        model_mean + 1.96*model_std,
        color='orange',
        alpha=0.3,
        label='95% confidence band',
    )

    if train_threshold_pu_rate_examples is not None:
        plt.scatter(
            train_threshold_pu_rate_examples[:, 1],
            train_threshold_pu_rate_examples[:, 2],
            alpha=0.3,
            c='red',
            label='Training points',
        )
    if val_threshold_pu_rate_examples is not None:
        plt.scatter(
            val_threshold_pu_rate_examples[:, 1],
            val_threshold_pu_rate_examples[:, 2],
            alpha=0.3,
            c='blue',
            label='Validation points'
        )
    plt.xlabel('PU')
    plt.ylabel('Rate (Hz)')
    plt.legend(loc='upper left')
    plt.title(title)

    plt.savefig(f'{output_path}/{name}.png')
    plt.clf()
    

#TODO the known points pretty hard coded and shouldn't be
def make_threshold_model_plot(
        model,
        pu,
        name,
        output_path,
        known_points=None,
):
    thresholds_test = np.linspace(0.0, 200.0, num=500)
    x_test = np.append(
        thresholds_test.reshape((-1,1)),
        np.tile(pu, reps=500).reshape((-1,1)),
        axis=1,
    )

    model_mean, model_std = model.predict(x_test, return_std=True)

    plt.plot(
        thresholds_test,
        model_mean,
        label='Model',
        c='orange',
    )

    plt.fill_between(
        thresholds_test,
        model_mean - 1.96*model_std,
        model_mean + 1.96*model_std,
        color='orange',
        alpha = 0.3,
        label='95% confidence interval'
    )

    if known_points is not None:
        plt.scatter(
            known_points[:, 0],
            known_points[:, 1],
            alpha=0.3,
            c='red',
            label='Measured Rates'
        )

    plt.yscale('log')
    plt.xlabel('CICADA Score')
    plt.ylabel('Rate (Hz)')
    plt.legend(loc='upper right')
    plt.title(f'PU = {pu}')

    plt.savefig(f'{output_path}/{name}.png')
    plt.clf()

def main(args):
    console.log('Start')
    output_path = Path(args.output_path)
    output_path.mkdir(exist_ok=True, parents=True)
    
    with open(args.model_file, 'rb') as the_file:
        model = pickle.load(the_file)

    threshold_pu = None
    rate = None
    for cicada_threshold in cicada_thresholds:
        with h5py.File(args.input_file) as the_file:
            data = np.array(the_file[cicada_threshold])
        if threshold_pu is None:
            threshold_pu = data[:, :2]
            rate = data[:, 2:]
        else:
            threshold_pu = np.append(threshold_pu, data[:, :2], axis=0)
            rate = np.append(rate, data[:, 2:], axis=0)

    x_train, x_val, y_train, y_val = train_test_split(
        threshold_pu,
        rate,
        test_size=args.test_size,
        random_state=42,
    )

    console.log('Making CICADA Seed PU Model plots')
    for cicada_seed in cicada_thresholds:
        the_threshold = cicada_thresholds[cicada_seed]
        train_threshold_pu_rate_examples = np.append(x_train, y_train, axis=1)
        train_threshold_pu_rate_examples = train_threshold_pu_rate_examples[train_threshold_pu_rate_examples[:, 0] == the_threshold]
        val_threshold_pu_rate_examples = np.append(x_val, y_val, axis=1)
        val_threshold_pu_rate_examples = val_threshold_pu_rate_examples[val_threshold_pu_rate_examples[:, 0] == the_threshold]
        make_pu_model_plot(
            model=model,
            threshold=the_threshold,
            output_path=output_path,
            title=f'{cicada_seed}',
            name=f'{cicada_seed}_pu_rate_model',
            train_threshold_pu_rate_examples=train_threshold_pu_rate_examples,
            val_threshold_pu_rate_examples=val_threshold_pu_rate_examples
        )

    console.log('Making Threshold Plots at different PU')
    make_threshold_model_plot(
        model=model,
        pu=59.7,
        name='Run392991_ActualPU_Rates',
        output_path=output_path,
        known_points = np.array( #SUPER hard coded...
            [
                [50.0, 8332.1],
                [60.0, 3579.92],
                [70.0, 1641.90],
                [80.0, 758.0],
                [90.0, 350.90],
                [110.0, 79.63],
                [125.0, 27.0],
                [135.0, 14.0],
            ]
        )
    )

    make_threshold_model_plot(
        model=model,
        pu=62.0,
        name='PU62_Model',
        output_path=output_path
    )
    
    make_threshold_model_plot(
        model=model,
        pu=64.0,
        name='PU64_Model',
        output_path=output_path
    )

    make_threshold_model_plot(
        model=model,
        pu=68.0,
        name='PU68_Model',
        output_path=output_path
    )
    console.log('Done')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Make plots for the threshold/pileup/rate model from OMS')

    parser.add_argument(
        'model_file',
        nargs='?',
        help='Finalized model from the fit',
    )

    parser.add_argument(
        'input_file',
        nargs='?',
        help='Model with inputs used to train the file'
    )

    parser.add_argument(
        '--output_path',
        nargs='?',
        default='./',
        help='Place to put plots'
    )
    parser.add_argument(
        '--test_size',
        nargs='?',
        default=0.4,
        help='Size of the set used in testing when the model was made',
        type=float,
    )

    args = parser.parse_args()

    main(args)
