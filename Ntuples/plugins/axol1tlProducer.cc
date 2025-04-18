#include <memory>
#include <iostream>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "DataFormats/L1Trigger/interface/BXVector.h"

#include "DataFormats/L1Trigger/interface/Muon.h"
#include "DataFormats/L1Trigger/interface/Tau.h"
#include "DataFormats/L1Trigger/interface/Jet.h"
#include "DataFormats/L1Trigger/interface/EGamma.h"
#include "DataFormats/L1Trigger/interface/EtSum.h"

#include <string>

#include "ap_fixed.h"
#include "hls4ml/emulator.h"


typedef ap_fixed<18, 13> inputtype;
typedef std::array<ap_fixed<10, 7, AP_RND_CONV, AP_SAT>, 8> resulttype;  //v3
typedef ap_ufixed<18, 14> losstype;
typedef std::pair<resulttype, losstype> pairtype;

class axol1tlProducer : public edm::one::EDProducer<>
{
public:
  explicit axol1tlProducer(const edm::ParameterSet&);
  ~axol1tlProducer() override=default;

private:
  void beginJob() override {};
  void produce(edm::Event&, const edm::EventSetup&) override;
  void endJob() override {};

  static constexpr int nMuons = 4;
  static constexpr int nJets = 10;
  static constexpr int nEGs = 4;
  static constexpr int nETSums = 1;
  static constexpr int nModelInput = (nMuons + nJets + nEGs + nETSums)*3;
  
  template<typename T>
  void fillParticles(
		     inputtype (&ADModelInput)[nModelInput],
		     uint& ADInputIndex,
		     const edm::Handle<T>& particleHandle,
		     const int maxParticles);
  
  template<typename template_result_type, typename template_loss_type>
  template_loss_type model_prediction();
  
  
  std::string modelName;
  edm::EDGetTokenT<BXVector<l1t::Muon>> muonToken;
  edm::EDGetTokenT<BXVector<l1t::Tau>> tauToken;
  edm::EDGetTokenT<BXVector<l1t::Jet>> jetToken;
  edm::EDGetTokenT<BXVector<l1t::EGamma>> egToken;
  edm::EDGetTokenT<BXVector<l1t::EtSum>> etSumToken;

  hls4mlEmulator::ModelLoader loader;
  std::shared_ptr<hls4mlEmulator::Model> model;
};

axol1tlProducer::axol1tlProducer(const edm::ParameterSet& iConfig):
  modelName(iConfig.getParameter<std::string>("modelName")),
  muonToken(consumes<BXVector<l1t::Muon>>(iConfig.getParameter<edm::InputTag>("muonToken"))),
  tauToken(consumes<BXVector<l1t::Tau>>(iConfig.getParameter<edm::InputTag>("tauToken"))),
  jetToken(consumes<BXVector<l1t::Jet>>(iConfig.getParameter<edm::InputTag>("jetToken"))),
  egToken(consumes<BXVector<l1t::EGamma>>(iConfig.getParameter<edm::InputTag>("egToken"))),
  etSumToken(consumes<BXVector<l1t::EtSum>>(iConfig.getParameter<edm::InputTag>("etSumToken"))),
  loader(hls4mlEmulator::ModelLoader(modelName))
{
  model = loader.load_model();
  produces<float>("AXOScore");
}

template<typename template_result_type, typename template_loss_type>
template_loss_type axol1tlProducer::model_prediction() {
  std::pair<template_result_type, template_loss_type> ADModelResult;
  model->read_result(&ADModelResult);
  return ADModelResult.second;
}

void axol1tlProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  edm::Handle<BXVector<l1t::Muon>> muonHandle;
  iEvent.getByToken(muonToken, muonHandle);

  edm::Handle<BXVector<l1t::Tau>> tauHandle;
  iEvent.getByToken(tauToken, tauHandle);

  edm::Handle<BXVector<l1t::Jet>> jetHandle;
  iEvent.getByToken(jetToken, jetHandle);

  edm::Handle<BXVector<l1t::EGamma>> egHandle;
  iEvent.getByToken(egToken, egHandle);

  edm::Handle<BXVector<l1t::EtSum>> etSumHandle;
  iEvent.getByToken(etSumToken, etSumHandle);

  inputtype fillzero = 0.0;
  inputtype ADModelInput[nModelInput];
  pairtype ADModelResult;

  uint ADInputIndex = 0;
  int usedEtSums = 0;
  //Okay, now we need to fill the model input, and get start filling it
  //First step, search the EtSum BX Vector for the MET
  // and fill that into the first three entries
  for(uint i = 0; i < etSumHandle->size(0); ++i)
    {
      const auto& theSum = etSumHandle->at(0, i);
      if(theSum.getType() == l1t::EtSum::EtSumType::kMissingEt){
	ADModelInput[ADInputIndex] = (inputtype)(theSum.hwPt() / 2); //Factor of 2 present in AXO emulator.
	//Quoth the AXO emulator:
	//"have to do hwPt/2 in order to match original et inputs"
	ADModelInput[ADInputIndex+1] = (inputtype)theSum.hwEta(); //always 0
	ADModelInput[ADInputIndex+2] = (inputtype)theSum.hwPhi();
	usedEtSums += 1;
	ADInputIndex += 3;
	break;
      }
    }

  //TODO: Huge amounts of this code is repeated except for the type
  // It could be abstracted pretty easily
  
  //Fill in any missing information with 0's
  //Shouldn't exist, but not a terrible precaution I guess
  while(usedEtSums < nETSums)
    {
      ADModelInput[ADInputIndex] = (inputtype) 0;
      ADModelInput[ADInputIndex+1] = (inputtype) 0; //always 0
      ADModelInput[ADInputIndex+2] = (inputtype) 0;
      usedEtSums += 1;
      ADInputIndex += 3;
    }

  //Fill any remaining particles in the standard style
  fillParticles<BXVector<l1t::EGamma>>(ADModelInput, ADInputIndex, egHandle, nEGs);
  fillParticles<BXVector<l1t::Muon>>(ADModelInput, ADInputIndex, muonHandle, nMuons);
  fillParticles<BXVector<l1t::Jet>>(ADModelInput, ADInputIndex, jetHandle, nJets);

  //Okay, this gets a bit tricky, because we now need to change around the types
  
  model->prepare_input(ADModelInput);
  model->predict();
  //model->read_result(&ADModelResult);
  std::unique_ptr<float> score_output;
  if(modelName == "GTADModel_v3" || modelName == "GTADModel_v4") {
    using case_result_type = std::array<ap_fixed<10, 7, AP_RND_CONV, AP_SAT>, 8>;
    score_output = std::make_unique<float>(model_prediction<case_result_type, losstype>().to_float());
  } else {
    using case_result_type = ap_fixed<18, 14, AP_RND_CONV, AP_SAT>;
    score_output = std::make_unique<float>(model_prediction<case_result_type, losstype>().to_float());
  }

  //This is how the axo emulator reads this
  //float score = ADModelResult.second.to_float() * 16.0;
  (*score_output) *= 16.0; //AXO factor present in the emulator.


  //std::unique_ptr<float> score_output = std::make_unique<float>(score);
  //std::cout<<"AXO Producer, Model Name: "<<modelName<<std::endl;
  //std::cout<<"Model Input: ";
  //for(int i = 0; i < nModelInput; ++i) std::cout<<ADModelInput[i]<<" ";
  //std::cout<<std::endl;
  //std::cout<<"Score: "<<*score_output<<std::endl;

  iEvent.put(std::move(score_output), "AXOScore");
 }

template<typename T>
void axol1tlProducer::fillParticles(
				    inputtype (&ADModelInput)[nModelInput],
				    uint& ADInputIndex,
				    const edm::Handle<T>& particleHandle,
				    const int maxParticles)
{
  int usedParticles = 0;
  //fill in as many particles as we have
  for(uint i = 0; i<std::min(particleHandle->size(0), (uint)maxParticles); ++i)
    {
      const auto& particle = particleHandle->at(0, i);
      ADModelInput[ADInputIndex] = (inputtype) (particle.hwPt() / 2);
      //Factor of 2 is constant in hardware ETs in AXO emulator
      ADModelInput[ADInputIndex+1] = (inputtype) particle.hwEta();
      ADModelInput[ADInputIndex+2] = (inputtype) particle.hwPhi();

      usedParticles += 1;
      ADInputIndex += 3;
    }

  while(usedParticles < maxParticles)
    {
      ADModelInput[ADInputIndex] = (inputtype) 0;
      ADModelInput[ADInputIndex+1] = (inputtype) 0;
      ADModelInput[ADInputIndex+2] = (inputtype) 0;

      usedParticles += 1;
      ADInputIndex += 3;
    }
}

DEFINE_FWK_MODULE(axol1tlProducer);
