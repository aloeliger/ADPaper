### Instructions for CICADA Paper and Ntuple Setup

#### CMSSW Setup
```
# Start by getting a CMSSW release
cmsrel CMSSW_15_1_0_pre1 # I recommend this one as the earliest with AXO and CICADA models
cd CMSSW_15_1_0_pre1/src/
cmsenv && git cms-init

#checkout relevant packages
git cms-addpkg DPGAnalysis/L1TNanoAOD L1Trigger/L1TCaloLayer1 L1Trigger/L1TGlobal

#Checkout the CICADA Nano Changes
git cms-rebase-topic -u aloeliger:CICADA_Nano
#If this causes merge conflicts, please fix them, and/or let me know

#Now checkout this repository
git clone git@github.com:aloeliger/ADPaper.git

#build everything
scram b -j 8
```

#### Running L1Nano with CICADA

The basic config should be under `ADPaper/Ntuples/python/l1nano_AD.py`

It is designed to be run on RAW or GEN-SIM-RAW

It can be run on data with:

```
cmsRun ADPaper/Ntuples/Python/l1nano_AD.py inputFiles=<YOUR INPUT FILE> isData=True outputFile=<YOUR OUTPUT FILE>
```

It can be run on MC with:

```
cmsRun ADPaper/Ntuples/Python/l1nano_AD.py inputFiles=<YOUR INPUT FILE> outputFile=<YOUR OUTPUT FILE>
```
