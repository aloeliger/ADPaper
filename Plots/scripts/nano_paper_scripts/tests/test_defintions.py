import pytest
import nano_paper_scripts.src.definitions as definitions

from .test_sample import low_stat_data_sample

def test_add_L1HT(low_stat_data_sample):
    definitions.add_L1HT(low_stat_data_sample)
    all_columns = low_stat_data_sample.df.GetColumnNames()
    assert('L1_HT' in all_columns)

def test_add_L1MET(low_stat_data_sample):
    definitions.add_L1MET(low_stat_data_sample)
    all_columns = low_stat_data_sample.df.GetColumnNames()
    assert('L1_MET' in all_columns)

def test_add_L1EG_sum_variable(low_stat_data_sample):
    definitions.add_L1EG_sum_variable(low_stat_data_sample)
    all_columns = low_stat_data_sample.df.GetColumnNames()
    assert('L1EG_pt_sum' in all_columns)

def test_get_unprescaled_trigger_list():
    unprescaled_triggers = definitions.get_unprescaled_trigger_list()
    assert('L1_SingleJet180' in unprescaled_triggers)

def test_make_pure_event_filter_string():
    unprescaled_triggers = [
        'L1_SingleJet180',
        'L1_SingleMu22',
    ]
    pure_event_filter_string = definitions.make_pure_event_filter_string(unprescaled_triggers)
    assert(
        pure_event_filter_string == 'L1_SingleJet180 == 0 && L1_SingleMu22 == 0'
    )

def test_add_pure_event_variable(low_stat_data_sample):
    definitions.add_pure_event_variable(low_stat_data_sample)
    all_columns = low_stat_data_sample.df.GetColumnNames()
    assert('pure_event' in all_columns)
    
def test_add_all_values(low_stat_data_sample):
    definitions.add_all_values(low_stat_data_sample)
    all_columns = low_stat_data_sample.df.GetColumnNames()
    assert('L1_HT' in all_columns)
    assert('L1_MET' in all_columns)
    assert('L1EG_pt_sum' in all_columns)
