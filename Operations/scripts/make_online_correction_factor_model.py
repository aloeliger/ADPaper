import pickle
import argparse
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.gaussian_process import GaussianProcessRegressor
import sklearn.gaussian_process as gp
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge

from rich.console import Console

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

def make_correction_factor_plot(
        model,
        train_inputs,
        train_factors,
        val_inputs,
        val_factors,
        name,
        output_path,
):
    model_inputs = np.linspace(0.0, 200.0, num=500)
    #model_mean, model_std = model.predict(model_inputs.reshape((-1,1)), return_std=True)
    model_mean = model.predict(model_inputs.reshape((-1,1)))

    plt.plot(
        model_inputs,
        model_mean,
        label='Correction factor model',
        c='orange'
    )
    # plt.fill_between(
    #     model_inputs,
    #     model_mean - 1.96*model_std,
    #     model_mean + 1.96*model_std,
    #     color='orange',
    #     alpha=0.3,
    #     label='95% confidence band',
    # )

    plt.scatter(
        train_inputs,
        train_factors,
        alpha=0.3,
        c='red',
        label='Training correction factors'
    )

    plt.scatter(
        val_inputs,
        val_factors,
        alpha=0.3,
        c='blue',
        label='Validation correction factors'
    )

    plt.xlabel('CICADA score')
    plt.ylabel('Correction factor to online')
    plt.title("Correction factors to OMS measurements")

    plt.legend(loc='upper right')
    
    plt.savefig(f'{output_path}/{name}.png')
    plt.clf()

def main(args):
    output_path = Path(args.output_path)
    output_path.mkdir(exist_ok=True, parents=True)
    
    with open(args.threshold_pu_rate_model, 'rb') as the_file:
        oms_model = pickle.load(the_file)

    with open(args.rate_table_model, 'rb') as the_file:
        rate_table_model = pickle.load(the_file)

    threshold_inputs = np.array(list(cicada_thresholds.values()))
    oms_model_inputs = np.append(
        threshold_inputs.reshape((-1, 1)),
        np.tile(args.run_PU, reps=len(threshold_inputs)).reshape((-1,1)),
        axis=1
    )
    rate_table_model_inputs = threshold_inputs.reshape((-1,1))

    oms_model_predictions = oms_model.predict(oms_model_inputs).reshape((-1,))
    rate_table_model_predictions = (rate_table_model.predict(rate_table_model_inputs).reshape((-1,)) * 1000.0)

    console.print(oms_model_predictions)
    console.print(rate_table_model_predictions)
    
    correction_factors =  oms_model_predictions /  rate_table_model_predictions #factor of 1000 because the rate table reports in kHz, the oms model in Hz


    best_loss = None
    best_model = None

    alphas = [
        0.001,
        0.01,
        0.1,
        1.0,
        10.0,
        100.0
    ]


    x_train, x_val, y_train, y_val = train_test_split(
        threshold_inputs.reshape((-1,1)),
        correction_factors.reshape((-1,1)),
        test_size = 0.3,
        random_state=42
    )
    for alpha in alphas:
        correction_factor_model = Pipeline(
            [
                (
                    'scaling',
                    StandardScaler()
                ),
                (
                    'model',
                    Ridge(alpha=alpha)
                )
            ]
        )

        correction_factor_model.fit(x_train, y_train)

        error = mean_squared_error(y_val, correction_factor_model.predict(x_val))

        if best_loss is None or error < best_loss:
            best_loss = error
            best_model = correction_factor_model
        console.print(f'RMS: {np.sqrt(error)}')

    make_correction_factor_plot(
        best_model,
        x_train,
        y_train,
        x_val,
        y_val,
        name='correction_factor_modeling',
        output_path=output_path
    )

    with open(f'{output_path}/correction_factor_model.pkl', 'wb') as the_file:
        pickle.dump(best_model, the_file)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Make a model based on the oms model and rate table model to correct the rate table model (or rate table) to true online rates')

    parser.add_argument(
        'threshold_pu_rate_model',
        nargs='?',
        help='Model fit to OMS threshold/PU data'
    )

    parser.add_argument(
        'rate_table_model',
        nargs='?',
        help='Model fit to OMS threshold/PU data'
    )

    parser.add_argument(
        'run_PU',
        nargs='?',
        type=float,
        help='PU of the run to evaluate the threshold/PU/rate model at'
    )

    parser.add_argument(
        '--output_path',
        nargs='?',
        help='Place to put the final model and plots',
        default='./'
    )

    args = parser.parse_args()

    main(args)
