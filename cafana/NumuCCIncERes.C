#include <numeric>

#include "TCanvas.h"
#include "TH1.h"
#include "TH2.h"
#include "TProfile.h"

#include "CAFAna/Core/Spectrum.h"
#include "CAFAna/Core/Var.h"
#include "CAFAna/Core/SpectrumLoader.h"

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
    {"TrueVtxZ",                {"True vertex z (cm)", ana::Binning::Simple(100, -200, 800),   kTrueVtxZ}},

    {"TrueLepEndX",             {"True lepton endpoint x (cm)", ana::Binning::Simple(140, -700, 700),  kTrueLepEndX}},
    {"TrueLepEndY",             {"True lepton endpoint y (cm)", ana::Binning::Simple(140, -700, 700),  kTrueLepEndY}},
    {"TrueLepEndZ",             {"True lepton endpoint z (cm)", ana::Binning::Simple(220, -700, 1500),  kTrueLepEndZ}},

    {"RecoMuonVtxX",            {"Muon candidate track vertex x (cm)", ana::Binning::Simple(80, -400, 400), kMuonCandVtxX}},
    {"RecoMuonVtxY",            {"Muon candidate track vertex y (cm)", ana::Binning::Simple(100, -250, 250), kMuonCandVtxY}},
    {"RecoMuonVtxZ",            {"Muon candidate track vertex z (cm)", ana::Binning::Simple(60, 350, 950), kMuonCandVtxZ}},

    {"MuCandLen",               {"Muon candidate track length (cm)", ana::Binning::Simple(70, 0, 700), kMuonCandLen}},
    {"NonMuCandTotalTrkEvis",   {"Sum of non-muon-candidate track visible energy (GeV)", ana::Binning::Simple(60, 0, 3), kNonMuonCandTotalTrkVisE}},
    {"ShowerTotalEvis",         {"Sum of shower visible energy (GeV)", ana::Binning::Simple(60, 0, 3), kTotalShwVisE}},
    {"NonMuTotalEvis",          {"Sum of non-#mu track and shower visible energy (GeV)", ana::Binning::Simple(60, 0, 3), kLoudHadEVis}},
//    {"NonMuTotalEvis",          {"Sum of non-#mu track and shower visible energy (GeV)", ana::Binning::Simple(60, 0, 3), kRecoHadVisE}},

    {"NTracks",                 {"Track multiplicity", ana::Binning::Simple(15, 0, 15), kNTracks}},

    // 2D plots
    {"TrueVtxXY",               {"True vertex x (cm)", ana::Binning::Simple(140, -700, 700),  kTrueVtxX,
                                "True vertex y (cm)",  ana::Binning::Simple(140, -700, 700), kTrueVtxY}},
    {"TrueVtxZX",               {"True vertex z (cm)", ana::Binning::Simple(140, -700, 700),  kTrueVtxZ,
                                "True vertex x (cm)",  ana::Binning::Simple(140, -700, 700), kTrueVtxX}},
    {"TrueVtxZY",               {"True vertex z (cm)", ana::Binning::Simple(140, -700, 700),  kTrueVtxZ,
                                "True vertex y (cm)",  ana::Binning::Simple(140, -700, 700), kTrueVtxY}},

    {"MuLenVsTrueEmu", {"True muon energy (GeV)",           ana::Binning::Simple(50, 0, 5), kTrueMuE,
                        "Muon candidate track length (cm)", ana::Binning::Simple(35, 0, 700), kMuonCandLen}},

    {"EmuVsTrueEmu", {"True muon energy (GeV)",  ana::Binning::Simple(30, 0, 3), kTrueMuE,
                      "Reco muon energy (GeV)",   ana::Binning::Simple(20, 0, 2), kRecoEmuFromTrkLen}},

    {"EmuResidVsTrueEmu", {"True muon energy (GeV)",           ana::Binning::Simple(50, 0, 5), kTrueMuE,
                           "(Reco E_{#mu} - true E_{#mu}) / true E_{#mu}", ana::Binning::Simple(41, -1, 1), (kRecoEmuFromTrkLen - kTrueMuE)/kTrueMuE}},

};

const std::map<std::string, ana::Cut>  CUTS
{
    {"NoCut",                   ana::kNoCut},
    {"RecoCont",                kHasAllContainedEnergy},
    {"NTracks",                 kNTracks > 0},
    {"NTracks+RecoCont",        (kNTracks > 0) && kHasAllContainedEnergy},
    {"NTracks+RecoCont+Signal", (kNTracks > 0) && kHasAllContainedEnergy && (kHasTrueMu && kIsVtxContained && kIsOutputContained)}
//    {"NumuReco",      kHasMuCandTrack},
//    {"NumuReco+RecoCont", kHasMuCandTrack && kHasAllContainedEnergy},
//    {"NumuReco+RecoCont+Signal", kHasMuCandTrack && kHasAllContainedEnergy && (kHasTrueMu && kIsVtxContained && kIsOutputContained)}
};

// ------------------------------------------------------------------------


// ------------------------------------------------------------------------
// ------------------------------------------------------------------------

void NumuCCIncERes(const std::string & inputCAF, const std::string & outdir)
{
  ana::SpectrumLoader loader(inputCAF);

  std::map<std::string, ana::Spectrum> spectra;

  for (const auto & cutPair : CUTS)
  {
    for (const auto &axisPair: VARS_TO_PLOT)
      spectra.emplace(std::piecewise_construct,
                      std::forward_as_tuple(axisPair.first + "_" + cutPair.first),
                      std::forward_as_tuple(loader, axisPair.second, cutPair.second));
  }

  loader.Go();

  TCanvas c;
  for (const auto & specPair : spectra)
  {
    const ana::Spectrum & spec = specPair.second;
    c.Clear();

    if (spec.NDimensions() == 1)
      spec.ToTH1(spec.POT())->DrawCopy("hist");
    else if (spec.NDimensions() == 2)
      spec.ToTH2(spec.POT())->DrawCopy("colz");
    c.SaveAs((outdir + "/" + specPair.first + ".png").c_str());
    c.SaveAs((outdir + "/" + specPair.first + ".root").c_str());
  }

  if (spectra.count("EmuResidVsTrueEmu_NTracks+RecoCont+Signal") > 0)
  {
    ana::Spectrum & spec = spectra.at("EmuResidVsTrueEmu_NTracks+RecoCont+Signal");
    TH2 * h2 = spec.ToTH2(spec.POT());
    TProfile * prof = h2->ProfileX("_pfx", 1, -1, "s");

    TH1D h("resol", "resol", prof->GetNbinsX(), prof->GetXaxis()->GetXmin(), prof->GetXaxis()->GetXmax());
    h.SetTitle(";True muon energy (GeV); RMS of (Reco E_{#mu} - True E_{#mu})/True E_{#mu}");
    for (int bin = 1; bin <= h.GetNbinsX(); bin++)
    {
      h.SetBinContent(bin, prof->GetBinError(bin));
      h.SetBinError(bin, 0);
    }

    c.Clear();
    h.SetMarkerStyle(20);
    h.Draw("p");
    c.SaveAs((outdir + "/EmuResol.png").c_str());
  }
}
