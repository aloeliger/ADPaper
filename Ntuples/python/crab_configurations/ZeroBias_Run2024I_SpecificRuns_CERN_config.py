from CRABClient.UserUtilities import config
import os
import datetime

today = datetime.date.today().strftime('%d%b%Y')
cmssw_base = os.getenv('CMSSW_BASE')

config = config()

config.General.requestName = f'AnomalyDetectionPaper2025_ZeroBias_Run2024I_SpecificRuns_CERN_{today}'
config.General.workArea = './crab'
config.General.transferOutputs = True

config.JobType.pluginName = 'Analysis'
config.JobType.psetName = f'{cmssw_base}/src/ADPaper/Ntuples/python/l1nano_AD.py'
config.JobType.pyCfgParams=['isData=True']
config.JobType.maxMemoryMB=4000

config.Data.inputDataset='/ZeroBias/Run2024I-v1/RAW'
config.Data.inputDBS = 'global'
config.Data.splitting='FileBased'
config.Data.unitsPerJob = 2
#config.Data.runRange='386478,386509,386554,386594,386604,386618,386640,386661,386668,386673,386679,386694,386704,386852,386864,386873,386885,386917,386924,386945,386951'
config.Data.runRange='386594,386604,386618,386640,386661,386668'
config.Data.publication=False
config.Data.outputDatasetTag = f'AnomalyDetectionPaper2025_ZeroBias_Run2024I_SpecificRuns_CERN_{today}'
config.Data.outLFNDirBase = '/store/group/phys_exotica/axol1tl/CICADANtuples'

config.Site.storageSite = 'T2_CH_CERN'
