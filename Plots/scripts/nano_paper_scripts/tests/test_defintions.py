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


def test_make_l1_trigger_event_filter_string():
    unprescaled_triggers = [
        'L1_SingleJet180',
        'L1_SingleMu22',
    ]

    unpure_event_filter_string = definitions.make_l1_trigger_event_filter_string(unprescaled_triggers)
    assert(
        unpure_event_filter_string == "!(L1_SingleJet180 == 0 && L1_SingleMu22 == 0)"
    )

def test_make_l1_trigger_variable(low_stat_data_sample):
    definitions.add_l1_trigger_variable(low_stat_data_sample)
    all_columns = low_stat_data_sample.df.GetColumnNames()
    assert('l1_event' in all_columns)


def test_get_scouting_trigger_list():
    scouting_trigger_list = definitions.get_scouting_trigger_list()
    assert('DST_PFScouting_JetHT' in scouting_trigger_list)

def test_get_HLT_trigger_list():
    HLT_trigger_list = definitions.get_HLT_trigger_list()
    assert('HLT_IsoMu27' in HLT_trigger_list)

def test_add_pure_HLT_event(low_stat_data_sample):
    definitions.add_pure_HLT_event_variable(low_stat_data_sample)
    all_columns = low_stat_data_sample.df.GetColumnNames()
    assert('pure_hlt_event' in all_columns)

def test_add_pure_scouting_event(low_stat_data_sample):
    definitions.add_pure_scouting_event_variable(low_stat_data_sample)
    all_columns = low_stat_data_sample.df.GetColumnNames()
    assert('pure_scouting_event' in all_columns)

def test_add_HLT_and_scouting_values(low_stat_data_sample):
    definitions.add_HLT_and_scouting_values(low_stat_data_sample)
    all_columns = low_stat_data_sample.df.GetColumnNames()
    assert('pure_hlt_event' in all_columns)
    assert('pure_scouting_event' in all_columns)

def test_get_collisions_runs():
    collisions_runs = definitions.get_collisions_runs()
    assert(isinstance(collisions_runs, list))
    
def test_make_collisions_runs_filter_string():
    list_of_runs = [1, 2]
    filter_string = definitions.make_collisions_runs_filter_string(list_of_runs)
    assert(filter_string == 'run == 1 || run == 2')

#TODO: This passes as long as the code runs. This should be updated
def test_make_collisions_runs_cuts(low_stat_data_sample):
    definitions.make_collisions_runs_cuts(low_stat_data_sample)

def test_get_v185_trigger_list():
    v185_runs = definitions.get_v185_trigger_list()
    assert(386531 in v185_runs)

def test_get_v189_trigger_list():
    v189_runs = definitions.get_v189_trigger_list()
    assert(386769 in v189_runs)

#TODO: This passes as long as the code runs. This should be updated
def test_make_v185_runs_cuts(low_stat_data_sample):
    definitions.make_v185_runs_cuts(low_stat_data_sample)

#TODO: This passes as long as the code runs. This should be updated
def test_make_v189_runs_cuts(low_stat_data_sample):
    definitions.make_v189_runs_cuts(low_stat_data_sample)
