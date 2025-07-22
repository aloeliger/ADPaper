from .sample import NanoSample
from .config import Configuration

def add_L1HT(the_sample: NanoSample):
    HT_Function = """
    int L1_HT = 0;
    try{
       for(int i = 0; i < nL1EtSum; ++i) {
          if(L1EtSum_bx.at(i) == 0 && L1EtSum_etSumType.at(i) == 1) L1_HT += L1EtSum_pt.at(i);
       }
       return L1_HT;
    }
    catch(const std::runtime_error& e) {
       return -999;
    }
    """

    the_sample.df = the_sample.df.Define("L1_HT", HT_Function)

def add_L1MET(the_sample: NanoSample):
    MET_Function = """
    int L1_MET = 0;
    try{
       for(int i = 0; i < nL1EtSum; ++i) {
          if(L1EtSum_bx.at(i) == 0 && L1EtSum_etSumType.at(i) == 2) L1_MET += L1EtSum_pt.at(i);
       }
       return L1_MET;
    }
    catch(const std::runtime_error& e) {
       return -999;
    }
    """

    the_sample.df = the_sample.df.Define("L1_MET", MET_Function)

def add_L1EG_sum_variable(the_sample: NanoSample):
    eg_pt_sum_function = """
    float pt_sum = 0.0;
    try {
       for(int i = 0; i < nL1EmulEG; ++i) {
          pt_sum += L1EmulEG_pt.at(i);
       }
       return pt_sum;
    }
    catch (const std::runtime_error& e) {
       return -999.f;
    }
    """

    the_sample.df = the_sample.df.Define('L1EG_pt_sum', eg_pt_sum_function)

def get_unprescaled_trigger_list():
    current_config = Configuration.GetConfiguration().configs
    unprescaled_trigger_list = current_config['unprescaled triggers']
    #unprescaled_trigger_list = current_config['All_L1_Triggers']
    return unprescaled_trigger_list

def get_scouting_trigger_list(is_mc=False):
    current_config = Configuration.GetConfiguration().configs
    if is_mc: #TODO No real scouting paths in our MC? This is likely bad configuration. This is a stop gap to guess what MC might end up in MC... but may have bad overlap with AD bits!
        scouting_trigger_list = ['Dataset_ScoutingPFRun3']
    else:
        scouting_trigger_list = current_config['All_Scouting_Triggers']
    return scouting_trigger_list

def get_HLT_trigger_list():
    current_config = Configuration.GetConfiguration().configs
    HLT_trigger_list = current_config['All_HLT_Triggers']
    return HLT_trigger_list

def get_v185_trigger_list():
    current_config = Configuration.GetConfiguration().configs
    v185_list = current_config['v185_collisions']
    return v185_list

def get_v189_trigger_list():
    current_config = Configuration.GetConfiguration().configs
    v189_list = current_config['v189_collisions']
    return v189_list

def get_collisions_runs():
    current_config = Configuration.GetConfiguration().configs
    collisions_runs = current_config['collisions_runs']
    return collisions_runs

def make_pure_event_filter_string(list_of_triggers):
    filter_string = ''
    for trigger in list_of_triggers:
        filter_string += f'{trigger} == 0 && '
    filter_string = filter_string[:-4]
    return filter_string

def make_l1_trigger_event_filter_string(list_of_triggers):
    return '!('+make_pure_event_filter_string(list_of_triggers)+')'

def add_pure_event_variable(the_sample: NanoSample):
    unprescaled_trigger_list = get_unprescaled_trigger_list()
    filter_string = make_pure_event_filter_string(unprescaled_trigger_list)
    the_sample.df = the_sample.df.Define('pure_event', filter_string)

def add_pure_scouting_event_variable(the_sample: NanoSample, is_mc: bool = False):
    scouting_trigger_list = get_scouting_trigger_list(is_mc=is_mc)
    filter_string = make_pure_event_filter_string(scouting_trigger_list)
    the_sample.df = the_sample.df.Define('pure_scouting_event', filter_string)

def add_pure_HLT_event_variable(the_sample: NanoSample):
    hlt_trigger_list = get_HLT_trigger_list()
    filter_string = make_pure_event_filter_string(hlt_trigger_list)
    the_sample.df = the_sample.df.Define('pure_hlt_event', filter_string)
    
def add_l1_trigger_variable(the_sample: NanoSample):
    unprescaled_trigger_list = get_unprescaled_trigger_list()
    filter_string = make_l1_trigger_event_filter_string(unprescaled_trigger_list)
    the_sample.df = the_sample.df.Define('l1_event', filter_string)

def make_collisions_runs_filter_string(list_of_runs: list[int]):
    filter_string = ''
    for run in list_of_runs:
        filter_string += f'run == {run} || '
    filter_string = filter_string[:-4]
    return filter_string

def make_collisions_runs_cuts(the_sample: NanoSample):
    list_of_runs = get_collisions_runs()
    collisions_cut_string = make_collisions_runs_filter_string(list_of_runs)
    the_sample.df = the_sample.df.Filter(collisions_cut_string)

def make_v185_runs_cuts(the_sample: NanoSample):
    list_of_runs = get_v185_trigger_list()
    collisions_cut_string = make_collisions_runs_filter_string(list_of_runs)
    the_sample.df = the_sample.df.Filter(collisions_cut_string)

def make_v189_runs_cuts(the_sample: NanoSample):
    list_of_runs = get_v189_trigger_list()
    collisions_cut_string = make_collisions_runs_filter_string(list_of_runs)
    the_sample.df = the_sample.df.Filter(collisions_cut_string)

def add_all_values(the_sample: NanoSample):
    add_L1HT(the_sample)
    add_L1MET(the_sample)
    add_L1EG_sum_variable(the_sample)
    add_pure_event_variable(the_sample)
    add_l1_trigger_variable(the_sample)

def add_HLT_and_scouting_values(the_sample: NanoSample, is_mc=False):
    add_pure_HLT_event_variable(the_sample)
    add_pure_scouting_event_variable(the_sample, is_mc=is_mc)
