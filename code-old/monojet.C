/*
Background Simulation b mu mu. 
Implemented 2 muons with opposite signs and 1 b-jet requirement.

root -l examples/monojet.C
*/

#include "TH1.h"
#include "TSystem.h"

#ifdef __CLING__
R__LOAD_LIBRARY(libDelphes)
#include "classes/DelphesClasses.h"
#include "external/ExRootAnalysis/ExRootTreeReader.h"
#include "external/ExRootAnalysis/ExRootResult.h"
#endif
#include <typeinfo>
#include <fstream>
#include <iostream>
#include <string>

//------------------------------------------------------------------------------

struct MyPlots
{
  TH1 *fJet1PT;
  TH1 *fJet2PT;
  TH1 *fMuon1PT;
  TH1 *fMuon2PT;
  TH1 *fMissingET;
  TH1 *fInvM;
  TH1 *fInvbM;
};

//------------------------------------------------------------------------------

class ExRootResult;
class ExRootTreeReader;

//------------------------------------------------------------------------------

void BookHistograms(ExRootResult *result, MyPlots *plots)
{
  // book histogram for pts
  plots->fJet1PT = result->AddHist1D(
    "leading_jet_pt", "leading-jet P_{T}",
    "leading jet P_{T}, GeV", "number of jets",
    50, 0.0, 1000.0);
    
  plots->fJet2PT = result->AddHist1D(
    "subleading_jet_pt", "leading-jet P_{T}",
    "leading jet P_{T}, GeV", "number of jets",
    50, 0.0, 1000.0);  
    

  // book histograms for missing ET and invariant mass
  plots->fMissingET = result->AddHist1D(
    "missing_et", "Missing E_{T}",
    "Missing E_{T}, GeV", "number of events",
    50, 0.0, 1000.0);
    
 }


//------------------------------------------------------------------------------

void AnalyseEvents(ExRootTreeReader *treeReader, Long64_t allEntries, MyPlots *plots, string output, string output1)
{
  TClonesArray *branchParticle = treeReader->UseBranch("Particle");
  TClonesArray *branchJet = treeReader->UseBranch("Jet");
  TClonesArray *branchMissingET = treeReader->UseBranch("MissingET");

  cout << "** Chain contains " << allEntries << " events" << endl;

  GenParticle *par1, *par2, *p1;
  Jet *jet, *jet1, *jet2;
  MissingET *met;
  Long64_t entry;

  
  

  int eventcount = 0; // count passed events

  // Define the output files
  ofstream PTJ;
  PTJ.open(output);
  ofstream ET;
  ET.open(output1);


  int Nmrec, Nmtr; // number of reconstructed and truth level muons
  int pdg1; // PDG numbers of mother


  // Loop over all events
  for(entry = 0; entry < allEntries; ++entry)
  {
    // Load selected branches with data from specified event
    treeReader->ReadEntry(entry);

            // Plot various pts
            if(branchJet->GetEntriesFast()>0)
            
            {
            jet1 = (Jet*) branchJet->At(0);
            
            if((jet1->PT)>130)
            {
            plots->fJet1PT->Fill(jet1->PT);
//            PTJ << jet1->PT <<","<<jet1->Eta<<","<<jet1->Phi<<","<<"1"<<endl;
//            PTJ << jet1->PT << ","<< jet2->PT <<endl;
            if(branchJet->GetEntriesFast()==2)
            {
            jet2 = (Jet*) branchJet->At(1);
            if((jet2->PT)>25)
            {
//            plots->fJet2PT->Fill(jet2->PT);
            PTJ << jet1->PT <<","<<jet1->Eta<<","<<jet1->Phi<<endl;
            }
            }
            }
            }
            // Analyse missing ET
            if(branchMissingET->GetEntriesFast() > 0)
            {
              met = (MissingET*) branchMissingET->At(0);
              plots->fMissingET->Fill(met->MET);
              ET << met->MET << endl;
            }
 
      }
      
   
  PTJ.close();
  ET.close();
  

  double eff = static_cast<double>(eventcount) / allEntries;
  cout << "Number of events that passed: " << eventcount << endl;
  cout << "Efficiency: " << eff << endl;
}

//------------------------------------------------------------------------------

void PrintHistograms(ExRootResult *result, MyPlots *plots)
{
  result->Print("pdf");
}

//------------------------------------------------------------------------------

void GenerateCSV(const char *inputFile, Long64_t allEntries, string output, string output1)
{
  TChain *chain = new TChain("Delphes");
  chain->Add(inputFile);

  ExRootTreeReader *treeReader = new ExRootTreeReader(chain);
  ExRootResult *result = new ExRootResult();

  if(allEntries == 0)
  {
    allEntries = treeReader->GetEntries();
  }

  MyPlots *plots = new MyPlots;

  BookHistograms(result, plots);

  AnalyseEvents(treeReader, allEntries, plots, output, output1);

  PrintHistograms(result, plots);

  result->Write("results.root");

  cout << "** Exiting..." << endl;

  delete plots;
  delete result;
  delete treeReader;
  delete chain;

}


//------------------------------------------------------------------------------

void monojet()
{
  gSystem->Load("libDelphes");

  const char *inputFile;

  inputFile = "/its/home/ck373/Desktop/scratch/MG5_aMC_v2_6_3_2/susyfullDelphes1/Events/run_01/tag_1_delphes_events.root";
  string output = "/its/home/ck373/Desktop/scratch/MG5_aMC_v2_6_3_2/susyfullDelphes1/susyfulldelphes.csv";
  string output1 = "/its/home/ck373/Desktop/scratch/MG5_aMC_v2_6_3_2/susyfullDelphes1/susy1_ET.csv";
 

  Long64_t allEntries =0 ; // Number of events, 0 = all



  GenerateCSV(inputFile, allEntries, output, output1);

}

//------------------------------------------------------------------------------
