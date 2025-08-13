import argparse
import pickle
import numpy as np
import json

from rich.console import Console

console = Console()

def make_overall_rate_predictor(
        rate_table_model,
        oms_model,
        correction_factor_model,
        derivation_run_pileup,
        gamma=1.0e-6
):
    def make_overall_rate_prediction(thresholds, pileup):
        rate_table_prediction = rate_table_model.predict(thresholds).reshape((-1,))
        oms_model_input = np.append(
            thresholds,
            np.tile(pileup, reps = len(thresholds)).reshape((-1, 1)),
            axis=1,
        )
        oms_denom_model_input = np.append(
            thresholds,
            np.tile(derivation_run_pileup, reps=len(thresholds)).reshape((-1, 1)),
            axis=1
        )
        pileup_factor = (oms_model.predict(oms_model_input) / (oms_model.predict(oms_denom_model_input)+gamma)).reshape((-1,))
        correction_factor = correction_factor_model.predict(thresholds).reshape((-1,))
        return rate_table_prediction * pileup_factor * correction_factor
    return make_overall_rate_prediction

def make_pure_rate_predictor(
        rate_table_model,
        oms_model,
        correction_factor_model,
        pure_frac_model,
        derivation_run_pileup,
        gamma=1.0e-6
):
    def make_pure_rate_prediction(thresholds, pileup):
        rate_table_prediction = rate_table_model.predict(thresholds).reshape((-1,))
        pure_fraction = pure_frac_model.predict(thresholds).reshape((-1,))
        oms_model_input = np.append(
            thresholds,
            np.tile(pileup, reps = len(thresholds)).reshape((-1, 1)),
            axis=1,
        )
        oms_denom_model_input = np.append(
            thresholds,
            np.tile(derivation_run_pileup, reps=len(thresholds)).reshape((-1, 1)),
            axis=1
        )
        pileup_factor = (
            oms_model.predict(oms_model_input) / (oms_model.predict(oms_denom_model_input)+gamma)
        ).reshape((-1,))
        correction_factor = correction_factor_model.predict(thresholds).reshape((-1,))

        return rate_table_prediction * pileup_factor * correction_factor * pure_fraction
    return make_pure_rate_prediction


def main(args):
    with open(args.rate_table_model_path, 'rb') as the_file:
        rate_table_model = pickle.load(the_file)

    with open(args.pure_frac_model_path, 'rb') as the_file:
        pure_frac_model = pickle.load(the_file)

    with open(args.oms_model_path, 'rb') as the_file:
        oms_model = pickle.load(the_file)

    with open(args.correction_factor_model_path, 'rb') as the_file:
        correction_factor_model = pickle.load(the_file)

    overall_rate_predictor = make_overall_rate_predictor(
        rate_table_model,
        oms_model,
        correction_factor_model,
        derivation_run_pileup=59.7
    )

    pure_rate_predictor = make_pure_rate_predictor(
        rate_table_model,
        oms_model,
        correction_factor_model,
        pure_frac_model,
        derivation_run_pileup=59.7
    )

    #thresholds = [50.0, 60.0, 70.0, 80.0, 90.0]
    thresholds = np.linspace(0.0, 256.0, (256*4)+1).reshape((-1,1))
    #thresholds = np.array(thresholds).reshape((-1, 1))

    # console.print(overall_rate_predictor(
    #     thresholds, args.pileup
    # ))
    # console.print(pure_rate_predictor(
    #     thresholds, args.pileup
    # ))

    overall_rates = overall_rate_predictor(thresholds, args.pileup)
    pure_rates = pure_rate_predictor(thresholds, args.pileup)

    console.print(overall_rates)
    console.print(pure_rates)
    
    rate_dict = {
        'overall': {},
        'pure': {},
    }

    for index, threshold in enumerate(thresholds.reshape((-1,))):
        overall_rate = overall_rates[index]
        pure_rate = pure_rates[index]
        rate_dict['overall'][threshold] = overall_rate
        rate_dict['pure'][threshold] = pure_rate

    with open (args.output_file, 'w') as the_file:
        json.dump(rate_dict, the_file, indent=3)
        
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Use rate table models, OMS models, and correction factor models to create a dedicated rate table')

    parser.add_argument(
        'rate_table_model_path',
        nargs='?',
        help='Path to model derived on rate table for predicting overall rates'
    )
    parser.add_argument(
        'pure_frac_model_path',
        nargs='?',
        help='Path to model derived on rate table for predicting pure fraction'
    )
    parser.add_argument(
        'oms_model_path',
        nargs='?',
        help='Path to model derived on OMS lumi data for doing threshold/PU -> rate predictions'
    )
    parser.add_argument(
        'correction_factor_model_path',
        nargs='?',
        help='Path to model derived on OMS lumi data and rate model data to correct for offline->online predictions'
    )
    parser.add_argument(
        'derivation_run_pileup',
        nargs='?',
        type=float,
        help='Pileup used in run used for the original rate table derivation'
    )

    parser.add_argument(
        'pileup',
        nargs='?',
        help='pileup to derive the new rate table at'
    )

    parser.add_argument(
        '--output_file',
        nargs='?',
        help='Name of the output file to leave the final rate table'
    )

    args = parser.parse_args()

    main(args)
