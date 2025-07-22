import pytest
import nano_paper_scripts.src.sample as samples

def test_Sample():
    sample = samples.NanoSample(["/hdfs/store/user/aloelige/ZeroBias/AnomalyDetectionPaper2025_ZeroBias_Run2024I_17Apr2025"])
    assert(sample.df is not None)

def test_construct_data_samples():
    data_samples = samples.construct_data_samples()
    assert(data_samples['RunI'] is not None)

def test_construct_mc_samples():
    mc_samples = samples.construct_mc_samples()
    assert(mc_samples["GluGluHToGG"] is not None)
    
def test_construct_data_samples_limited():
    data_samples = samples.construct_data_samples(limit_files=1)
    assert(data_samples['RunI'] is not None)
    assert(data_samples['RunI'].nFiles == 1)

@pytest.fixture
def low_stat_data_sample():
    data_samples = samples.construct_data_samples(limit_files=1)
    return data_samples['RunI']

@pytest.fixture
def low_stat_signal_sample():
    signal_samples = samples.construct_mc_samples(limit_files=1)
    return signal_samples['TT']

def test_JsonConfiguredNanoSample():
    sample = samples.JsonConfiguredNanoSample('configuration/sample_nano_paths.json')
    assert(sample.df is not None)

@pytest.fixture
def low_stat_collision_data_sample():
    sample = samples.JsonConfiguredNanoSample('configuration/sample_nano_paths.json')
    return sample

def test_construct_collisions_runs_trigger_files_only():
    the_sample_collection= samples.construct_collisions_runs_trigger_files_only()
    assert(the_sample_collection['RunI_collisions_only'].df is not None)
