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
from sklearn.ensemble import BaggingRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import SelectKBest, f_regression

from rich.console import Console

console = Console()

cicada_thresholds = [
    'L1_CICADA_VLoose',
    'L1_CICADA_Loose',
    'L1_CICADA_Medium',
    'L1_CICADA_Tight',
    'L1_CICADA_VTight',
    'L1_CICADA_VVTight',
    'L1_CICADA_VVVTight',
    'L1_CICADA_VVVVTight',
]

def main(args):
    output_path = Path(args.output_path)
    output_path.mkdir(exist_ok=True, parents=True)
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

    # kernels = [
    #     gp.kernels.ConstantKernel(1.0, constant_value_bounds=(1e-6, 1e6))*gp.kernels.RBF(1.0, length_scale_bounds=(1e-6, 1e6))+gp.kernels.WhiteKernel(1.0, noise_level_bounds=(1e-6, 1e6)),
    #     gp.kernels.ConstantKernel(1.0, constant_value_bounds=(1e-6, 1e6))*gp.kernels.Matern(1.0, length_scale_bounds=(1e-6, 1e6))+gp.kernels.WhiteKernel(1.0, noise_level_bounds=(1e-6, 1e6)),
    #     gp.kernels.ConstantKernel(1.0, constant_value_bounds=(1e-6, 1e6))*gp.kernels.RationalQuadratic(1.0, 1.0, length_scale_bounds=(1e-6, 1e6), alpha_bounds=(1e-6, 1e6))+gp.kernels.WhiteKernel(1.0, noise_level_bounds=(1e-6, 1e6))
    # ]

    # kernels = [
    #     gp.kernels.ConstantKernel(1.0, constant_value_bounds=(1e-6, 1e6))*gp.kernels.RBF(5.0, length_scale_bounds='fixed')+gp.kernels.WhiteKernel(1.0, noise_level_bounds=(1e-6, 1e6)),
    #     gp.kernels.ConstantKernel(1.0, constant_value_bounds=(1e-6, 1e6))*gp.kernels.RBF(10.0, length_scale_bounds='fixed')+gp.kernels.WhiteKernel(1.0, noise_level_bounds=(1e-6, 1e6)),
    #     gp.kernels.ConstantKernel(1.0, constant_value_bounds=(1e-6, 1e6))*gp.kernels.RBF(4.0, length_scale_bounds='fixed')+gp.kernels.WhiteKernel(1.0, noise_level_bounds=(1e-6, 1e6)),
    #     gp.kernels.ConstantKernel(1.0, constant_value_bounds=(1e-6, 1e6))*gp.kernels.RBF(8.0, length_scale_bounds='fixed')+gp.kernels.WhiteKernel(1.0, noise_level_bounds=(1e-6, 1e6)),
    # ]

    
    x_train, x_val, y_train, y_val = train_test_split(
        threshold_pu,
        rate,
        test_size=args.test_size,
        random_state=42,
    )

    console.log(x_train.shape)
    console.log(x_val.shape)
    console.log(y_train.shape)
    console.log(y_val.shape)
    
    best_loss = None
    best_model = None
    #for kernel in kernels:
    gpr = Pipeline(
        [
            (
                'scaling',
                StandardScaler()
            ),
            (
                'gpr',
                GaussianProcessRegressor(
                    n_restarts_optimizer=3,
                    #kernel=gp.kernels.ConstantKernel(1.0, constant_value_bounds=(1e-6, 1e6))*gp.kernels.RBF(1.0, length_scale_bounds=(1e-6, 1e6))+gp.kernels.WhiteKernel(1.0, noise_level_bounds=(1e-6, 1e6)),
                    #kernel=gp.kernels.ConstantKernel(1.0, constant_value_bounds=(1e-6, 1e6))*gp.kernels.Matern(1.0, length_scale_bounds=(1e-6, 1e6))+gp.kernels.WhiteKernel(1.0, noise_level_bounds=(1e-6, 1e6)),
                    kernel=gp.kernels.ConstantKernel(1.0, constant_value_bounds=(1e-6, 1e6))*gp.kernels.RationalQuadratic(1.0, 1.0, length_scale_bounds=(1e-6, 1e6), alpha_bounds=(1e-6, 1e6))+gp.kernels.WhiteKernel(1.0, noise_level_bounds=(1e-6, 1e6)),
                    #kernel=gp.kernels.DotProduct(1.0, sigma_0_bounds=(1e-6, 1e6)),
                    #kernel = kernel,
                    normalize_y=True
                )
            )
        ]
    )
    
    gpr.fit(x_train, y_train)
    
    val_predictions = gpr.predict(x_val)
    error = np.sqrt(mean_squared_error(y_val, val_predictions))
    console.log(f'Val Regression error: {error:.4f}')
    # if best_loss == None or error < best_loss:
    #     best_loss = error
    #     best_model = gpr
    best_model = gpr
    with open(f'{output_path}/threshold_pu_rate_model.pkl', 'wb') as the_file:
        pickle.dump(best_model, the_file)
            
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Make models from oms data')

    parser.add_argument(
        'input_file',
        nargs='?',
        help='File to read input information from'
    )

    parser.add_argument(
        '--output_path',
        nargs='?',
        help='Path to leave output models and plots',
        default='./',
    )

    parser.add_argument(
        '--test_size',
        nargs='?',
        help='Size of the test set, to control training time',
        type=float,
        default=0.4,
    )

    args = parser.parse_args()
    
    main(args)
