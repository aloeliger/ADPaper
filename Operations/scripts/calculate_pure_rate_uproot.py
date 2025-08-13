import ROOT
from rich.console import Console
from rich.table import Table
from rich.progress import track
import os
from tensorflow import keras
import numpy as np
import json
import pickle
from sklearn.model_selection import train_test_split
from sklearn.gaussian_process import GaussianProcessRegressor
import sklearn.gaussian_process as gp
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from pathlib import Path
import argparse
import matplotlib.pyplot as plt
import uproot

console = Console()

unprescaled_triggers = [
    "L1_DoubleEG8er2p5_HTT300er",
    "L1_DoubleEG8er2p5_HTT320er",
    "L1_DoubleEG_25_12_er2p5",
    "L1_DoubleEG_25_14_er2p5",
    "L1_DoubleEG_27_14_er2p5",
    "L1_DoubleEG_LooseIso20_LooseIso12_er1p5",
    "L1_DoubleEG_LooseIso22_12_er2p5",
    "L1_DoubleEG_LooseIso22_LooseIso12_er1p5",
    "L1_DoubleEG_LooseIso25_12_er2p5",
    "L1_DoubleEG_LooseIso25_LooseIso12_er1p5",
    #"L1_DoubleIsoTau26er2p1_Jet55_RmOvlp_dR0p5",
    #"L1_DoubleIsoTau26er2p1_Jet70_RmOvlp_dR0p5",
    "L1_DoubleIsoTau34er2p1",
    "L1_DoubleIsoTau35er2p1",
    "L1_DoubleIsoTau36er2p1",
    "L1_DoubleJet112er2p3_dEta_Max1p6",
    "L1_DoubleJet150er2p5",
    "L1_DoubleJet30er2p5_Mass_Min250_dEta_Max1p5",
    "L1_DoubleJet30er2p5_Mass_Min300_dEta_Max1p5",
    "L1_DoubleJet30er2p5_Mass_Min330_dEta_Max1p5",
    "L1_DoubleJet_110_35_DoubleJet35_Mass_Min800",
    "L1_DoubleLLPJet40",
    "L1_DoubleLooseIsoEG22er2p1",
    "L1_DoubleLooseIsoEG24er2p1",
    "L1_DoubleMu0_Upt15_Upt7",
    "L1_DoubleMu0_Upt6_IP_Min1_Upt4",
    "L1_DoubleMu0_dR_Max1p6_Jet90er2p5_dR_Max0p8",
    "L1_DoubleMu0er1p4_SQ_OS_dR_Max1p4",
    "L1_DoubleMu18er2p1_SQ",
    "L1_DoubleMu3_OS_er2p3_Mass_Max14_DoubleEG7p5_er2p1_Mass_Max20",
    "L1_DoubleMu3_SQ_ETMHF40_HTT60er",
    "L1_DoubleMu3_SQ_ETMHF40_Jet60er2p5_OR_DoubleJet40er2p5",
    "L1_DoubleMu3_SQ_ETMHF50_HTT60er",
    "L1_DoubleMu3_SQ_ETMHF50_Jet60er2p5",
    "L1_DoubleMu3_SQ_ETMHF50_Jet60er2p5_OR_DoubleJet40er2p5",
    "L1_DoubleMu3_SQ_ETMHF60_Jet60er2p5",
    "L1_DoubleMu3_SQ_HTT220er",
    "L1_DoubleMu4er2p0_SQ_OS_dR_Max1p6",
    "L1_DoubleMu4p5_SQ_OS_dR_Max1p2",
    "L1_DoubleMu4p5er2p0_SQ_OS_Mass_7to18",
    "L1_DoubleMu4p5er2p0_SQ_OS_Mass_Min7",
    "L1_DoubleMu5_OS_er2p3_Mass_8to14_DoubleEG3er2p1_Mass_Max20",
    "L1_DoubleMu5_SQ_EG9er2p5",
    "L1_DoubleMu8_SQ",
    "L1_DoubleMu9_SQ",
    "L1_DoubleMu_15_5_SQ",
    "L1_DoubleMu_15_7",
    "L1_DoubleMu_15_7_SQ",
    "L1_DoubleTau70er2p1",
    "L1_ETM150",
    "L1_ETMHF100",
    "L1_ETMHF100_HTT60er",
    "L1_ETMHF110",
    "L1_ETMHF110_HTT60er",
    "L1_ETMHF120",
    "L1_ETMHF120_HTT60er",
    "L1_ETMHF130",
    "L1_ETMHF130_HTT60er",
    "L1_ETMHF140",
    "L1_ETMHF150",
    "L1_ETMHF90",
    "L1_ETMHF90_HTT60er",
    "L1_ETMHF90_SingleJet60er2p5_dPhi_Min2p1",
    "L1_ETMHF90_SingleJet60er2p5_dPhi_Min2p6",
    "L1_ETT2000",
    "L1_HTT200_SingleLLPJet60",
    "L1_HTT240_SingleLLPJet70",
    "L1_HTT280er",
    "L1_HTT280er_QuadJet_70_55_40_35_er2p5",
    "L1_HTT320er",
    "L1_HTT320er_QuadJet_70_55_40_40_er2p5",
    "L1_HTT320er_QuadJet_80_60_er2p1_45_40_er2p3",
    "L1_HTT320er_QuadJet_80_60_er2p1_50_45_er2p3",
    "L1_HTT360er",
    "L1_HTT400er",
    "L1_HTT450er",
    "L1_LooseIsoEG22er2p1_IsoTau26er2p1_dR_Min0p3",
    "L1_LooseIsoEG22er2p1_Tau70er2p1_dR_Min0p3",
    "L1_LooseIsoEG24er2p1_IsoTau27er2p1_dR_Min0p3",
    "L1_Mu12er2p3_Jet40er2p3_dR_Max0p4_DoubleJet40er2p3_dEta_Max1p6",
    "L1_Mu18er2p1_Tau24er2p1",
    "L1_Mu18er2p1_Tau26er2p1",
    "L1_Mu18er2p1_Tau26er2p1_Jet55",
    "L1_Mu18er2p1_Tau26er2p1_Jet70",
    "L1_Mu20_EG10er2p5",
    "L1_Mu22er2p1_IsoTau32er2p1",
    "L1_Mu22er2p1_IsoTau34er2p1",
    "L1_Mu22er2p1_IsoTau40er2p1",
    "L1_Mu22er2p1_Tau70er2p1",
    "L1_Mu3er1p5_Jet100er2p5_ETMHF40",
    "L1_Mu3er1p5_Jet100er2p5_ETMHF50",
    "L1_Mu6_DoubleEG12er2p5",
    "L1_Mu6_DoubleEG15er2p5",
    "L1_Mu6_DoubleEG17er2p5",
    "L1_Mu6_HTT240er",
    "L1_Mu6_HTT250er",
    "L1_Mu7_EG20er2p5",
    "L1_Mu7_EG23er2p5",
    "L1_Mu7_LooseIsoEG20er2p5",
    "L1_Mu7_LooseIsoEG23er2p5",
    "L1_QuadJet_95_75_65_20_DoubleJet_75_65_er2p5_Jet20_FWD3p0",
    "L1_SingleEG36er2p5",
    "L1_SingleEG38er2p5",
    "L1_SingleEG40er2p5",
    "L1_SingleEG42er2p5",
    "L1_SingleEG45er2p5",
    "L1_SingleEG60",
    "L1_SingleIsoEG30er2p1",
    "L1_SingleIsoEG30er2p5",
    "L1_SingleIsoEG32er2p1",
    "L1_SingleIsoEG32er2p5",
    "L1_SingleIsoEG34er2p5",
    "L1_SingleJet180",
    "L1_SingleJet180er2p5",
    "L1_SingleJet200",
    "L1_SingleJet43er2p5_NotBptxOR_3BX",
    "L1_SingleJet46er2p5_NotBptxOR_3BX",
    "L1_SingleMu22",
    "L1_SingleMu25",
    "L1_SingleMuOpen_er1p1_NotBptxOR_3BX",
    "L1_SingleMuOpen_er1p4_NotBptxOR_3BX",
    "L1_SingleMuShower_Nominal",
    "L1_SingleMuShower_Tight",
    "L1_SingleTau120er2p1",
    "L1_SingleTau130er2p1",
    #"L1_TripleEG_18_17_8_er2p5",
    #"L1_TripleEG_18_18_12_er2p5",
    "L1_TripleJet_100_80_70_DoubleJet_80_70_er2p5",
    "L1_TripleJet_105_85_75_DoubleJet_85_75_er2p5",
    "L1_TripleJet_95_75_65_DoubleJet_75_65_er2p5",
    "L1_TripleMu3_SQ",
    "L1_TripleMu_4SQ_2p5SQ_0_OS_Mass_Max12",
    "L1_TripleMu_5SQ_3SQ_0_DoubleMu_5_3_SQ_OS_Mass_Max9",
    "L1_TripleMu_5_3_3",
    "L1_TripleMu_5_3_3_SQ",
    "L1_TripleMu_5_3p5_2p5_DoubleMu_5_2p5_OS_Mass_5to17",
    "L1_TripleMu_5_4_2p5_DoubleMu_5_2p5_OS_Mass_5to17",
    "L1_TripleMu_5_5_3",
    "L1_TwoMuShower_Loose"
]

def make_pure_event_filter_string(list_of_triggers):
    filter_string = ''
    for trigger in list_of_triggers:
        filter_string += f'{trigger} == 0 && '
    filter_string = filter_string[:-4]
    return filter_string

def add_pure_event_variable(the_df, unprescaled_triggers):
    return the_df.Define('pure_event', make_pure_event_filter_string(unprescaled_triggers))

def get_predictions_for_df(the_df, the_model):
    console.log('Getting inputs')
    df_info = the_df.AsNumpy(['Regions_iphi', 'Regions_ieta', 'Regions_et'])
    Regions_iphi = df_info['Regions_iphi']
    Regions_iphi = np.array([list(x) for x in Regions_iphi]).reshape((-1, 18, 14))

    Regions_ieta = df_info['Regions_ieta']
    Regions_ieta = np.array([list(x) for x in Regions_ieta]).reshape((-1, 18, 14))

    Regions_et = df_info['Regions_et']
    Regions_et = np.array([list(x) for x in Regions_et]).reshape((-1, 18, 14))

    #console.log(Regions_iphi.shape)
    #console.log(Regions_ieta.shape)
    #console.log(Regions_et.shape)

    outputs = np.zeros((len(Regions_et), 18, 14))
    outputs[:, Regions_iphi, Regions_ieta] = Regions_et
    #console.log(outputs.shape)

    outputs = outputs.reshape((-1,252))
    predictions = the_model.predict(outputs)
    return predictions

def convert_eff_to_rate(eff, nBunches = 2544):
    return eff * (float(nBunches) * 11425e-3)

def derive_rate_table_model(scores, output_location):
    console.log(f'Deriving rate table models')
    console.log(f'Maximum score: {np.max(scores):.4f}')
    console.log(f'Minimum score: {np.min(scores):.4f}')
    console.log(f'Mean score: {np.mean(scores):.4f}')
    console.log(f'Std-Dev score: {np.std(scores):.4f}')
    thresholds = np.linspace(0.25, 256.0, num=(256*4))

    #Okay, let's do the bootstrapping procedure
    n_trials = 2500
    score_samples = []
    for i in track(range(n_trials), description="Bootstrapping rate table..."):
        sample = np.random.choice(scores, size=len(scores), replace=True)
        score_samples.append(sample)
    eff_samples = {}
    for threshold in track(thresholds, description="Eff per threshold..."):
        eff_sample = []
        for score_sample in score_samples:
            eff = np.sum(score_sample >= threshold) / len(score_sample)
            eff_sample.append(eff)
        eff_samples[threshold] = np.array(eff_sample)

    #Let's make the rate table 
    rate_table = {}
    rate_table['overall'] = {}
    rate_table['overall']['central_value'] = {}
    rate_table['overall']['uncertainty'] = {}

    rate_uncertainties = []
    mean_rates = []
    thresholds = []
    for threshold in eff_samples:
        rates = convert_eff_to_rate(eff_samples[threshold])

        thresholds.append(threshold)
        mean_rate = np.mean(rates)
        mean_rates.append(mean_rate)
        std_rate = np.std(rates)
        rate_uncertainties.append(std_rate)
        
        rate_table['overall']['central_value'][threshold] = mean_rate
        rate_table['overall']['uncertainty'][threshold] = std_rate
    #console.print(rate_table)

    console.log('Making regression model for overall rates')
    #Let's make a model to fit to the
    mean_rate_uncertainty = np.mean(rate_uncertainties)
    console.log(f'Mean rate uncertainty {mean_rate_uncertainty:.4f}')
    mean_rates = np.array(mean_rates)
    thresholds = np.array(thresholds)

    train_thresholds, val_thresholds, train_rates, val_rates, train_rate_uncertainties, val_rate_uncertainties = train_test_split(
        thresholds,
        mean_rates,
        rate_uncertainties,
        random_state=42,
        test_size = 0.3/1.0
    )
    console.log(f'Num training points: {len(train_thresholds)}')
    console.log(f'Num val points: {len(val_thresholds)}')

    kernels = [
        gp.kernels.ConstantKernel(1.0, constant_value_bounds=(1e-6, 1e6))*gp.kernels.RBF(1.0, length_scale_bounds=(1e-6, 1e6))+gp.kernels.WhiteKernel(1.0, noise_level_bounds=(1e-6, 1e6)),
        gp.kernels.ConstantKernel(1.0, constant_value_bounds=(1e-6, 1e6))*gp.kernels.Matern(1.0, length_scale_bounds=(1e-6, 1e6))+gp.kernels.WhiteKernel(1.0, noise_level_bounds=(1e-6, 1e6)),
        gp.kernels.ConstantKernel(1.0, constant_value_bounds=(1e-6, 1e6))*gp.kernels.RationalQuadratic(1.0, 1.0, length_scale_bounds=(1e-6, 1e6), alpha_bounds=(1e-6, 1e6))+gp.kernels.WhiteKernel(1.0, noise_level_bounds=(1e-6, 1e6))
        # 1.0 * gp.kernels.RBF(2.0, length_scale_bounds="fixed"),
        # 1.0 * gp.kernels.RBF(1.0, length_scale_bounds="fixed"),
        # 1.0 * gp.kernels.RBF(0.5, length_scale_bounds="fixed"),
        # 1.0 * gp.kernels.RBF(0.25, length_scale_bounds="fixed"),
        # 1.0 * gp.kernels.RBF(0.1, length_scale_bounds="fixed"),
        # 1.0 * gp.kernels.RBF(0.01, length_scale_bounds="fixed"),
    ]
    best_loss = None
    best_estimator = None
    for kernel in track(kernels, description='Optimizing kernel', console=console):
        gpr = Pipeline(
            [
                (
                    'scaling',
                    StandardScaler()
                ),
                (
                    'gpr',
                    GaussianProcessRegressor(
                        #alpha=(np.array(train_rate_uncertainties)**2)+1e-6,
                        n_restarts_optimizer=3,
                        #kernel=1.0 * gp.kernels.RBF(1.0, length_scale_bounds=(1e-6,1e6)),
                        #kernel=1.0 * gp.kernels.RationalQuadratic(1.0, 1.0, length_scale_bounds=(1e-6,1e6), alpha_bounds=(13-6, 1e6)),
                        kernel = kernel,
                        normalize_y = True
                    )
                )
            ]
        )
        
        
        gpr.fit(train_thresholds.reshape((-1,1)), train_rates.reshape((-1,1)))
        val_predictions = gpr.predict(val_thresholds.reshape((-1,1)))
        error = mean_squared_error(val_rates, val_predictions)
        if best_loss is None or error < best_loss:
            best_loss = error
            best_estimator = gpr
        console.log(f'Val Regression error: {error:.4f}')
    val_predictions = best_estimator.predict(val_thresholds.reshape((-1,1)))
    error = mean_squared_error(val_rates, val_predictions)
    train_error = mean_squared_error(train_rates, gpr.predict(train_thresholds.reshape((-1,1))))
    console.log(f'Mean rate: {np.mean(val_rates)}')
    console.log(f'Train Regression error: {train_error:.4f}')
    console.log(f'Val Regression error: {error:.4f}')

    plot_gpr_model(
        gpr,
        train_thresholds,
        train_rates,
        train_rate_uncertainties,
        val_thresholds,
        val_rates,
        val_rate_uncertainties,
        'Overall Rate',
        0.01,
        28000.0,
        'overall_rate_gpr_model',
        output_location,
    )
    
    # for i in range(256):
    #     prediction = gpr.predict([[i]])
    #     console.print(f'{i}: {prediction}')

    return rate_table, best_estimator

def derive_pure_rate_fractions(pure_scores, overall_scores, rate_table, output_location):
    #Okay. Time to bootstrap the pure fraction above certain thresholds
    thresholds = np.linspace(0.25, 256.0, num=(256*4))
    n_trials = 2500
    pure_score_samples = []
    for i in track(range(n_trials), description="Bootstrapping pure rates..."):
        sample = np.random.choice(pure_scores, size=len(overall_scores), replace=True)
        pure_score_samples.append(sample)
    pure_score_frac_samples = {}
    for threshold in track(thresholds, description="Purity per threshold..."):
        pure_score_frac_sample = []
        for pure_score_sample in pure_score_samples:
            denom = np.sum(overall_scores >= threshold)
            if denom == 0.0:
                frac = 0
            else:
                frac = np.sum(pure_score_sample >= threshold) / denom
            pure_score_frac_sample.append(frac)
        pure_score_frac_samples[threshold] = np.array(pure_score_frac_sample)

    rate_table['pure'] = {}
    rate_table['pure']['frac'] = {}
    rate_table['pure']['central_value'] = {}
    rate_table['pure']['uncertainty'] = {}

    frac_uncertainties = []
    mean_fracs = []
    thresholds = []
    for threshold in pure_score_frac_samples:
        fracs = pure_score_frac_samples[threshold]
        
        thresholds.append(threshold)
        mean_frac = np.mean(fracs)
        mean_fracs.append(mean_frac)

        std_frac = np.std(fracs)
        frac_uncertainties.append(std_frac)

        rate_table['pure']['frac'][threshold] = mean_frac
        rate_table['pure']['central_value'][threshold] = mean_frac * rate_table['overall']['central_value'][threshold]
        rate_table['pure']['uncertainty'][threshold] = std_frac * rate_table['pure']['central_value'][threshold]

    #We have the full rate table, let's get a model
    console.log('Making pure fraction regressor')
    mean_frac_uncertainty = np.mean(frac_uncertainties)
    mean_fracs = np.array(mean_fracs)
    thresholds = np.array(thresholds)

    train_thresholds, val_thresholds, train_fracs, val_fracs, train_frac_uncertainties, val_frac_uncertainties = train_test_split(
        thresholds,
        mean_fracs,
        frac_uncertainties,
        random_state=42,
        test_size=0.3/1.0
    )

    kernels = [
        gp.kernels.ConstantKernel(1.0, constant_value_bounds=(1e-6, 1e6))*gp.kernels.RBF(1.0, length_scale_bounds=(1e-6, 1e6))+gp.kernels.WhiteKernel(1.0, noise_level_bounds=(1e-6, 1e6)),
        gp.kernels.ConstantKernel(1.0, constant_value_bounds=(1e-6, 1e6))*gp.kernels.Matern(1.0, length_scale_bounds=(1e-6, 1e6))+gp.kernels.WhiteKernel(1.0, noise_level_bounds=(1e-6, 1e6)),
        gp.kernels.ConstantKernel(1.0, constant_value_bounds=(1e-6, 1e6))*gp.kernels.RationalQuadratic(1.0, 1.0, length_scale_bounds=(1e-6, 1e6), alpha_bounds=(1e-6, 1e6))+gp.kernels.WhiteKernel(1.0, noise_level_bounds=(1e-6, 1e6))
        # 1.0 * gp.kernels.RBF(1.0, length_scale_bounds="fixed"),
        # 1.0 * gp.kernels.RBF(0.5, length_scale_bounds="fixed"),
        # 1.0 * gp.kernels.RBF(0.25, length_scale_bounds="fixed"),
        # 1.0 * gp.kernels.RBF(0.1, length_scale_bounds="fixed"),
        # 1.0 * gp.kernels.RBF(0.01, length_scale_bounds="fixed"),
    ]
    best_loss = None
    best_estimator = None
    for kernel in track(kernels, description='Optimizing kernel', console=console):
        gpr = Pipeline(
            [
                (
                    'scaling',
                    StandardScaler()
                ),
                (
                    'gpr',
                    GaussianProcessRegressor(
                        #alpha=(np.array(train_frac_uncertainties)**2)+1e-6,
                        n_restarts_optimizer=3,
                        #kernel=1.0 * gp.kernels.RBF(1.0, length_scale_bounds=(1e-6,1e6)),
                        #kernel=1.0 * gp.kernels.RationalQuadratic(1.0, 1.0, length_scale_bounds=(1e-6,1e6), alpha_bounds=(13-6, 1e6)),
                        kernel=kernel,
                        normalize_y=True,
                    )
                )   ,
            ]
        )

        
        gpr.fit(train_thresholds.reshape((-1, 1)), train_fracs.reshape((-1, 1)))
        val_predictions = gpr.predict(val_thresholds.reshape((-1, 1)))
        error = mean_squared_error(val_fracs, val_predictions)
        if best_loss is None or error < best_loss:
            best_loss = error
            best_estimator=gpr
        console.log(f'Val Regression error: {error:.4f}')
    val_predictions = best_estimator.predict(val_thresholds.reshape((-1, 1)))
    error = mean_squared_error(val_fracs, val_predictions)
    train_error = mean_squared_error(train_fracs, gpr.predict(train_thresholds.reshape((-1,1))))

    console.log(f'Mean fraction: {np.mean(val_fracs)}')
    console.log(f'Train Regression Error: {train_error:.4f}')
    console.log(f'Val Regression error: {error:.4f}')

    plot_gpr_model(
        gpr,
        train_thresholds,
        train_fracs,
        train_frac_uncertainties,
        val_thresholds,
        val_fracs,
        val_frac_uncertainties,
        'Pure Fraction',
        0.01,
        1.0,
        'pure_fraction_gpr_model',
        output_location,
    )
    
    return rate_table, gpr

def generate_smooth_model_table(rate_model, frac_model):
    thresholds = np.linspace(0.0, 256.0, num=(256*4)+1).reshape((-1, 1))
    overall_rates = rate_model.predict(thresholds)
    fracs = frac_model.predict(thresholds)
    pure_rates = overall_rates * fracs
    rate_tables = {
        'overall_rates': {},
        'pure_fracs': {},
        'pure_rates': {},
    }

    for index in range(len(thresholds)):
        threshold = thresholds[index][0]
        rate_tables['overall_rates'][threshold] = overall_rates[index]
        rate_tables['pure_fracs'][threshold] = fracs[index]
        rate_tables['pure_rates'][threshold] = pure_rates[index]
    return rate_tables

def plot_gpr_model(
        gpr,
        x_train,
        y_train,
        y_unc_train,
        x_val,
        y_val,
        y_unc_val,
        y_axis_name,
        ylow,
        yhigh,
        name,
        output_location):
    thresholds = np.linspace(0.0, 256.0, num=(256*100)+1)
    gpr_mean, gpr_std = gpr.predict(thresholds.reshape((-1, 1)), return_std=True)
    plt.plot(
        thresholds,
        gpr_mean,
        label='Gaussian Process Regression',
        c='orange',
    )
    plt.fill_between(
        thresholds,
        gpr_mean - 1.96*gpr_std,
        gpr_mean + 1.96*gpr_std,
        color='orange',
        alpha=0.5,
        label='95% confidence bound'
    )
    plt.errorbar(
        x_train,
        y_train,
        y_unc_train,
        linestyle='None',
        c='red',
        marker='.',
        markersize=2,
        linewidth=0.1,
        label='Training Points'
    )
    plt.errorbar(
        x_val,
        y_val,
        y_unc_val,
        linestyle='None',
        c='blue',
        marker='.',
        markersize=2,
        linewidth=0.1,
        label='Val Points'
    )

    plt.xlabel('Thresholds')
    plt.ylabel(y_axis_name)
    plt.ylim(ylow, yhigh)
    plt.xlim(0.0, 200.0)
    plt.yscale('log')

    plt.legend()

    plt.savefig(f'{output_location}/{name}.png')
    plt.clf()

def make_score_histogram(scores, output_location, name):
    plt.hist(
        scores,
        range=(0.0,200.0),
        bins=50,
        density=True,
        histtype='stepfilled'
    )

    plt.xlabel('CICADA Scores')
    plt.ylabel('A.U.')
    #plt.yscale('log')
    #plt.xlim(0.0, 200.0)
    plt.title(name)
    plt.savefig(f'{output_location}/{name}.png')
    plt.clf()

def get_inputs(list_of_files):
    file_and_tree = [x+':Events' for x in list_of_files]
    #batch_num = 0
    branches_to_load = [
        'run',
        'CICADAUnpacked_CICADAScore'
    ] + unprescaled_triggers

    scores = []
    pure_scores = []
    used_branches = lambda x: x in branches_to_load
    for batch in track(uproot.iterate(file_and_tree, filter_name=used_branches), console=console):
        # arrays = batch.arrays(branches_to_load)
        scores += list(batch.CICADAUnpacked_CICADAScore)
        batch_unprescaled_triggers = np.zeros((len(batch.CICADAUnpacked_CICADAScore)))
        for unprescaled_trigger in unprescaled_triggers:
            batch_unprescaled_triggers += np.array(batch[unprescaled_trigger])
        batch_pure_events = batch_unprescaled_triggers == 0
        batch_pure_scores = list(np.array(batch.CICADAUnpacked_CICADAScore)[batch_pure_events])
        pure_scores += batch_pure_scores
        # console.log(batch.CICADAUnpacked_CICADAScore)
        # console.log(batch_pure_events)
        # console.log(batch_num)
        #batch_num+=1
    scores = np.array(scores)
    pure_scores = np.array(pure_scores)
    return scores, pure_scores

def main(args):
    console.log('Start')
    
    output_location = Path(args.output_location)
    output_location.mkdir(exist_ok=True, parents=True)
    
    #files_path = '/hdfs/store/user/aloelige/ZeroBias/Operations_ZeroBias_Run2025C_CERN_20May2025'
    #files_path = '/hdfs/store/user/aloelige/ZeroBias/Operations_ZeroBias_Run2025C_21May2025/'
    files_path = '/hdfs/store/user/aloelige/ZeroBias/Operations_ZeroBias_Run2025C_Run392991_05Jun2025'
    all_files = []
    file_index = 0
    for root, dirs, files in os.walk(files_path):
        for fileName in files:
            if args.file_dilation != 0 and file_index % args.file_dilation==0:
                file_index+=1
                continue
            file_index+=1
            the_file = f'{root}/{fileName}'
            all_files.append(the_file)

    total_scores, pure_scores = get_inputs(all_files)
    # event_inputs = event_inputs.reshape((-1,252))
    # pure_event_inputs = pure_event_inputs.reshape((-1,252))
    # #exit(0)
    
        
    # #Okay, well, they didn't save CICADA triggers, so we need to get the
    # #regions from data and rerun the model

    # the_model = keras.models.load_model("ADPaper/Ntuples/data/best-legacy-method")
    # total_entries = the_chain.GetEntries()
    # total_scores = the_model.predict(event_inputs).reshape((-1,))
    # pure_scores = the_model.predict(pure_event_inputs).reshape((-1,))

    # console.log(f'Unique runs considered: {unique_runs}')
    console.log(f'{len(total_scores)} total scores')
    console.log(f'{len(pure_scores)} pure scores')

    make_score_histogram(
        total_scores,
        output_location,
        name='overall_scores'
    )
    make_score_histogram(
        pure_scores,
        output_location,
        name='pure_scores'
    )
    
    rate_table, rate_model = derive_rate_table_model(total_scores, output_location)
    rate_table, frac_model = derive_pure_rate_fractions(pure_scores, total_scores, rate_table, output_location)

    with open(f"{output_location}/rate_table.json", "w") as the_file:
        json.dump(rate_table, the_file, indent=3)
    with open(f"{output_location}/overall_rate_model.pkl", "wb") as the_file:
        pickle.dump(rate_model, the_file)
    with open(f"{output_location}/pure_frac_model.pkl", "wb") as the_file:
        pickle.dump(frac_model, the_file)

    smooth_rate_table = generate_smooth_model_table(rate_model, frac_model)

    with open(f"{output_location}/model_table.json", "w") as the_file:
        json.dump(smooth_rate_table, the_file, indent=3)
    
    console.log('Done!')
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--output_location',
        nargs='?',
        default='pure_rate_modeling',
    )
    parser.add_argument(
        '--file_dilation',
        nargs='?',
        type=int,
        help='Dilation factor for number of files used',
        default=0,
    )

    args = parser.parse_args()
    main(args)
