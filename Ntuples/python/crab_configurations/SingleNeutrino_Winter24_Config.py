from CRABClient.UserUtilities import config
import os
import datetime

today = datetime.date.today().strftime('%d%b%Y')
cmssw_base = os.getenv('CMSSW_BASE')

config = config()

config.General.requestName = f'AnomalyDetectionPaper2025_SingleNeutrino_Winter24_{today}'

config.General.workArea = './crab'
config.General.transferOutputs = True

config.JobType.pluginName = 'Analysis'
config.JobType.psetName = f'{cmssw_base}/src/ADPaper/Ntuples/python/l1nano_AD.py'
config.JobType.maxMemoryMB=4000

config.Data.inputDataset='/SingleNeutrino_E-10-gun/Run3Winter24Digi-133X_mcRun3_2024_realistic_v9-v3/GEN-SIM-RAW'
config.Data.inputDBS = 'global'
config.Data.splitting = 'FileBased'
config.Data.unitsPerJob = 3
config.Data.publication = False
config.Data.outputDatasetTag = f'AnomalyDetectionPaper2025_SingleNeutrino_Winter24_{today}'

config.Site.storageSite = 'T2_US_Wisconsin'
