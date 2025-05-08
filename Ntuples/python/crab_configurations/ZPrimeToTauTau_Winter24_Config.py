from CRABClient.UserUtilities import config
import os
import datetime

today = datetime.date.today().strftime('%d%b%Y')
cmssw_base = os.getenv('CMSSW_BASE')

config = config()

config.General.requestName = f'AnomalyDetectionPaper2025_ZPrimeToTauTau_Winter24_{today}'

config.General.workArea = './crab'
config.General.transferOutputs = True

config.JobType.pluginName = 'Analysis'
config.JobType.psetName = f'{cmssw_base}/src/ADPaper/Ntuples/python/l1nano_AD.py'
config.JobType.maxMemoryMB=4000

config.Data.inputDataset='/ZprimeToTauTau_M-4000_TuneCP5_tauola_13p6TeV-pythia8/Run3Winter24Digi-133X_mcRun3_2024_realistic_v8-v2/GEN-SIM-RAW'
config.Data.inputDBS = 'global'
config.Data.splitting = 'FileBased'
config.Data.unitsPerJob = 1
config.Data.publication = False
config.Data.outputDatasetTag = f'AnomalyDetectionPaper2025_ZPrimeToTauTau_Winter24_{today}'

config.Site.storageSite = 'T2_US_Wisconsin'

