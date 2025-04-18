import FWCore.ParameterSet.Config as cms

axol1tlProducerv3 = cms.EDProducer(
    'axol1tlProducer',
    modelName=cms.string('GTADModel_v3'),
    #muonToken = cms.InputTag("simGmtStage2Digis"),
    muonToken = cms.InputTag("simGmtStage2Digis"),
    egToken = cms.InputTag("simCaloStage2Digis"),
    tauToken = cms.InputTag("simCaloStage2Digis"),
    jetToken = cms.InputTag("simCaloStage2Digis"),
    etSumToken = cms.InputTag("simCaloStage2Digis"),
)

axol1tlProducerv4 = cms.EDProducer(
    'axol1tlProducer',
    modelName=cms.string('GTADModel_v4'),
    #muonToken = cms.InputTag("simGmtStage2Digis"),
    muonToken = cms.InputTag("simGmtStage2Digis"),
    egToken = cms.InputTag("simCaloStage2Digis"),
    tauToken = cms.InputTag("simCaloStage2Digis"),
    jetToken = cms.InputTag("simCaloStage2Digis"),
    etSumToken = cms.InputTag("simCaloStage2Digis"),
)

axol1tlProducerv5 = cms.EDProducer(
    'axol1tlProducer',
    modelName=cms.string('GTADModel_v5'),
    #muonToken = cms.InputTag("simGmtStage2Digis"),
    muonToken = cms.InputTag("simGmtStage2Digis"),
    egToken = cms.InputTag("simCaloStage2Digis"),
    tauToken = cms.InputTag("simCaloStage2Digis"),
    jetToken = cms.InputTag("simCaloStage2Digis"),
    etSumToken = cms.InputTag("simCaloStage2Digis"),
)
