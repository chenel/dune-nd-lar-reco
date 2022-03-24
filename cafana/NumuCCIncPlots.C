#include <numeric>

#include "TCanvas.h"
#include "TH1.h"
#include "TH2.h"
#include "TLatex.h"
#include "TProfile.h"

#include "CAFAna/Core/Spectrum.h"
#include "CAFAna/Core/Var.h"
#include "CAFAna/Core/SpectrumLoader.h"

#include "PlotStyle.h"
#include "EnergyEstimatorCutsVars.h"


const ana::Var kLoudHadEVis([](const caf::SRProxy * sr) -> float
{
  float hadEVis = kNonMuonCandTotalTrkVisE(sr) + kTotalShwVisE(sr);
  if (kHasMuCandTrack(sr) && kIsOutputContained(sr) && (kHasTrueMu(sr) && kIsVtxContained(sr) && kHasAllContainedEnergy(sr))
      && hadEVis < 0.01)
    std::cout << "Near-zero EhadVis event: (" << sr->run << "," << sr->subrun << "," << sr->event << ") EhadVis = " << hadEVis << std::endl;
  return hadEVis;
});

// ------------------------------------
const std::map<std::string, ana::HistAxis> VARS_TO_PLOT
{
    {"TrueLepPDG",              {"True outgoing lepton PDG", ana::Binning::Simple(40, -20, 20),  kTrueLepPDG}},

    {"TrueVtxX",                {"True vertex x (cm)", ana::Binning::Simple(140, -700, 700),  kTrueVtxX}},
    {"TrueVtxY",                {"True vertex y (cm)", ana::Binning::Simple(140, -700, 700), kTrueVtxY}},
    {"TrueVtxZ",                {"True vertex z (cm)", ana::Binning::Simple(100, 200, 1200),   kTrueVtxZ}},

    {"TrueLepEndX",             {"True lepton endpoint x (cm)", ana::Binning::Simple(140, -700, 700),  kTrueLepEndX}},
    {"TrueLepEndY",             {"True lepton endpoint y (cm)", ana::Binning::Simple(140, -700, 700),  kTrueLepEndY}},
    {"TrueLepEndZ",             {"True lepton endpoint z (cm)", ana::Binning::Simple(220, -700, 1500),  kTrueLepEndZ}},

    {"RecoMuonVtxX",            {"Muon candidate track vertex x (cm)", ana::Binning::Simple(80, -400, 400), kMuonCandVtxX}},
    {"RecoMuonVtxY",            {"Muon candidate track vertex y (cm)", ana::Binning::Simple(100, -250, 250), kMuonCandVtxY}},
    {"RecoMuonVtxZ",            {"Muon candidate track vertex z (cm)", ana::Binning::Simple(60, 350, 950), kMuonCandVtxZ}},

    {"RecoMuonEndX",            {"Muon candidate track end x (cm)", ana::Binning::Simple(80, -400, 400), kMuonCandEndX}},
    {"RecoMuonEndY",            {"Muon candidate track end y (cm)", ana::Binning::Simple(100, -250, 250), kMuonCandEndY}},
    {"RecoMuonEndZ",            {"Muon candidate track end z (cm)", ana::Binning::Simple(60, 350, 950), kMuonCandEndZ}},

    {"MuCandLen",               {"Muon candidate track length (cm)", ana::Binning::Simple(70, 0, 700), kMuonCandLen}},
    {"NonMuCandTotalTrkEvis",   {"Sum of non-muon-candidate track visible energy (GeV)", ana::Binning::Simple(60, 0, 3), kNonMuonCandTotalTrkVisE}},
    {"ShowerTotalEvis",         {"Sum of shower visible energy (GeV)", ana::Binning::Simple(60, 0, 3), kTotalShwVisE}},
    //{"NonMuTotalEvis",          {"Sum of non-#mu track and shower visible energy (GeV)", ana::Binning::Simple(60, 0, 3), kLoudHadEVis}},
    {"NonMuTotalEvis",          {"Sum of non-#mu track and shower visible energy (GeV)", ana::Binning::Simple(60, 0, 3), kRecoHadVisE}},

    {"NTracks",                 {"Track multiplicity", ana::Binning::Simple(15, 0, 15), kNTracks}},
    {"NShowers",                {"Shower multiplicity", ana::Binning::Simple(15, 0, 15), kNShowers}},

    // 2D plots
    {"TrueVtxXY",               {"True vertex x (cm)", ana::Binning::Simple(140, -700, 700),  kTrueVtxX,
                                "True vertex y (cm)",  ana::Binning::Simple(140, -700, 700), kTrueVtxY}},
    {"TrueVtxZX",               {"True vertex z (cm)", ana::Binning::Simple(100, 200, 1200),  kTrueVtxZ,
                                "True vertex x (cm)",  ana::Binning::Simple(140, -700, 700), kTrueVtxX}},
    {"TrueVtxZY",               {"True vertex z (cm)", ana::Binning::Simple(100, 200, 1200),  kTrueVtxZ,
                                "True vertex y (cm)",  ana::Binning::Simple(140, -700, 700), kTrueVtxY}},

    {"MuLenVsTrueEmu", {"True muon energy (GeV)",           ana::Binning::Simple(50, 0, 5), kTrueMuE,
                        "Muon candidate track length (cm)", ana::Binning::Simple(35, 0, 700), kMuonCandLen}},

    {"EmuVsTrueEmu", {"True muon energy (GeV)",  ana::Binning::Simple(30, 0, 3), kTrueMuE,
                      "Reco muon energy (GeV)",   ana::Binning::Simple(20, 0, 2), kRecoEmuFromTrkLen}},

    {"EmuResidVsTrueEmu", {"True muon energy (GeV)",           ana::Binning::Simple(50, 0, 5), kTrueMuE,
                           "(E_{#mu}^{reco} - E_{#mu}^{true}) / E_{#mu}^{true}", ana::Binning::Simple(41, -1, 1), (kRecoEmuFromTrkLen - kTrueMuE)/kTrueMuE}},
    {"EnuResidVsTrueEnu", {"True neutrino energy (GeV)",           ana::Binning::Simple(50, 0, 5), kTrueEnu,
                          "(E_{#mu}^{reco} + E_{had}^{reco} - E_{#nu}^{true}) / E_{#nu}^{true}", ana::Binning::Simple(41, -1, 1), (kRecoEhadFromEhadVis + kRecoEmuFromTrkLen - kTrueEnu)/kTrueEnu}},
    {"EnuResidVsTrueYbj", {"True inelasticity y_{Bj}",   ana::Binning::Simple(21, 0, 1.05), kTrueInel,
                          "(E_{#mu}^{reco} + E_{had}^{reco} - E_{#nu}^{true}) / E_{#nu}^{true}", ana::Binning::Simple(41, -1, 1), (kRecoEhadFromEhadVis + kRecoEmuFromTrkLen - kTrueEnu)/kTrueEnu}},

};

const std::map<std::string, ana::Cut>  CUTS
{
    {"NoCut",                   ana::kNoCut},
    {"Signal",                  kIsSignal},
    {"RecoCont",                kHasAllContainedEnergy},
    {"RecoCont+Signal",         kHasAllContainedEnergy && kIsSignal},
    {"NTracks",                 kNTracks > 0},
    {"NTracks+Signal",          (kNTracks > 0) && kIsSignal},
    {"NTracks+RecoCont",        (kNTracks > 0) && kHasAllContainedEnergy},
    {"NTracks+RecoCont+Signal", (kNTracks > 0) && kHasAllContainedEnergy && kIsSignal}
//    {"NumuReco",      kHasMuCandTrack},
//    {"NumuReco+RecoCont", kHasMuCandTrack && kHasAllContainedEnergy},
//    {"NumuReco+RecoCont+Signal", kHasMuCandTrack && kHasAllContainedEnergy && (kHasTrueMu && kIsVtxContained && kIsOutputContained)}
};

// ------------------------------------------------------------------------


// ------------------------------------------------------------------------
// ------------------------------------------------------------------------

void NumuCCIncPlots(const std::string & inputCAF, const std::string & outdir, bool includeCutLabel=true, bool includeDUNEWIP=false)
{
  ana::SpectrumLoader loader(inputCAF);

  std::map<std::string, ana::Spectrum> spectra;

  std::map<std::string, std::string> specToCutMap;   // because this is easier than working backwards from the spectrum name...
  for (const auto & cutPair : CUTS)
  {
    for (const auto &axisPair: VARS_TO_PLOT)
    {
      std::string specName = axisPair.first + "_" + cutPair.first;
      spectra.emplace(std::piecewise_construct,
                      std::forward_as_tuple(specName),
                      std::forward_as_tuple(loader, axisPair.second, cutPair.second));
      specToCutMap.emplace(specName, cutPair.first);
    }
  }

  loader.Go();

  dunestyle::CherryInvertedPalette();

  gErrorIgnoreLevel = kWarning;  // I don't need to read all the "ROOT file has been created: ..." messages

  TCanvas c;
  for (const auto & specPair : spectra)
  {
    const ana::Spectrum & spec = specPair.second;
    c.Clear();

    TH1 * h = nullptr;
    if (spec.NDimensions() == 1)
      h = spec.ToTH1(spec.POT())->DrawCopy("hist");
    else if (spec.NDimensions() == 2)
    {
      h = spec.ToTH2(spec.POT())->DrawCopy("colz");
      h->SetZTitle("Events");
    }
    else
    {
      std::cout << " can't deal with " << spec.NDimensions() << "-D hist: " << specPair.first << std::endl;
      continue;
    }
    if (h)
      dunestyle::CenterTitles(h);

    std::unique_ptr<TLatex> cutLabel;
    if (includeCutLabel)
    {
      cutLabel = std::make_unique<TLatex>(0.9, 0.92, ("[" + specToCutMap.at(specPair.first) + "]").c_str());
      cutLabel->SetNDC();
      cutLabel->SetTextAlign(kVAlignBottom + kHAlignRight);
      cutLabel->Draw();
    }

    if (includeDUNEWIP)
      dunestyle::WIP(kHAlignLeft);

    c.SaveAs((outdir + "/" + specPair.first + ".png").c_str());
    c.SaveAs((outdir + "/" + specPair.first + ".root").c_str());
  }

  if (spectra.count("EmuResidVsTrueEmu_NTracks+RecoCont+Signal") > 0)
  {
    c.Clear();

    ana::Spectrum & spec = spectra.at("EmuResidVsTrueEmu_NTracks+RecoCont+Signal");
    TH2 * h2 = spec.ToTH2(spec.POT());
    TProfile * prof = h2->ProfileX("EmuResid_profx", 1, -1, "s");

    TH1D h("resol_emu", "resol", prof->GetNbinsX(), prof->GetXaxis()->GetXmin(), prof->GetXaxis()->GetXmax());
    h.SetTitle(";True muon energy (GeV); RMS of (E_{#mu}^{reco} - E_{#mu}^{true})/E_{#mu}^{true}");
    for (int bin = 1; bin <= h.GetNbinsX(); bin++)
    {
      h.SetBinContent(bin, prof->GetBinError(bin));
      h.SetBinError(bin, 0);
    }

    h.SetMarkerStyle(20);
    h.Draw("p");
    dunestyle::CenterTitles(&h);
    if (includeDUNEWIP)
      dunestyle::WIP(kHAlignLeft);
    c.SaveAs((outdir + "/EmuResol.png").c_str());
  }

  if (spectra.count("EnuResidVsTrueEnu_NTracks+RecoCont+Signal") > 0)
  {
    c.Clear();

    ana::Spectrum & spec = spectra.at("EnuResidVsTrueEnu_NTracks+RecoCont+Signal");
    TH2 * h2 = spec.ToTH2(spec.POT());
    TProfile * prof = h2->ProfileX("EnuResid_profx", 1, -1, "s");

    TH1D h("resol_Enu", "resol", prof->GetNbinsX(), prof->GetXaxis()->GetXmin(), prof->GetXaxis()->GetXmax());
    h.SetTitle(";True neutrino energy (GeV); RMS of (E_{#mu}^{reco} + E_{had}^{reco} - E_{#nu}^{true}) / E_{#nu}^{true}");
    for (int bin = 1; bin <= h.GetNbinsX(); bin++)
    {
      h.SetBinContent(bin, prof->GetBinError(bin));
      h.SetBinError(bin, 0);
    }

    h.SetMarkerStyle(20);
    h.Draw("p");
    dunestyle::CenterTitles(&h);
    if (includeDUNEWIP)
      dunestyle::WIP(kHAlignLeft);
    c.SaveAs((outdir + "/EnuResol.png").c_str());
  }

}
