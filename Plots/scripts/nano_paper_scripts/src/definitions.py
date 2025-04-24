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
    return unprescaled_trigger_list

def make_pure_event_filter_string(list_of_triggers):
    filter_string = ''
    for trigger in list_of_triggers:
        filter_string += f'{trigger} == 0 && '
    filter_string = filter_string[:-4]
    return filter_string

def add_pure_event_variable(the_sample: NanoSample):
    unprescaled_trigger_list = get_unprescaled_trigger_list()
    filter_string = make_pure_event_filter_string(unprescaled_trigger_list)
    the_sample.df = the_sample.df.Define('pure_event', filter_string)

def add_all_values(the_sample: NanoSample):
    add_L1HT(the_sample)
    add_L1MET(the_sample)
    add_L1EG_sum_variable(the_sample)
    add_pure_event_variable(the_sample)
