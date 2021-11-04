#include <numeric>

#include "TCanvas.h"
#include "TH1.h"
#include "TH2.h"
#include "TProfile.h"

#include "CAFAna/Core/Spectrum.h"
#include "CAFAna/Core/SpectrumLoader.h"

#include "EnergyEstimatorCutsVars.h"


// ------------------------------------
const std::map<std::string, ana::HistAxis> VARS_TO_PLOT
{
    {"MuCandLen",               {"Muon candidate track length (cm)", ana::Binning::Simple(70, 0, 700), kMuonCandLen}},
    {"NonMuCandTotalTrkEvis",   {"Sum of non-muon-candidate track visible energy (GeV)", ana::Binning::Simple(60, 0, 3), kNonMuonCandTotalTrkVisE}},
    {"ShowerTotalEvis",         {"Sum of shower visible energy (GeV)", ana::Binning::Simple(60, 0, 3), kTotalShwVisE}},

    {"NTracks",                 {"Track multiplicity", ana::Binning::Simple(15, 0, 15), kNTracks}},

    // 2D plots
    {"MuLenVsTrueEmu", {"True muon energy (GeV)",           ana::Binning::Simple(50, 0, 5), kTrueMuE,
                           "Muon candidate track length (cm)", ana::Binning::Simple(35, 0, 700), kMuonCandLen}},

    {"EmuVsTrueEmu", {"True muon energy (GeV)",           ana::Binning::Simple(50, 0, 5), kTrueMuE,
                         "Reco muon energy (GeV)", ana::Binning::Simple(50, 0, 5), kRecoEmuFromTrkLen}},

    {"EmuResidVsTrueEmu", {"True muon energy (GeV)",           ana::Binning::Simple(50, 0, 5), kTrueMuE,
                              "(Reco E_{#mu} - true E_{#mu}) / true E_{#mu}", ana::Binning::Simple(41, -1, 1), (kRecoEmuFromTrkLen - kTrueMuE)/kTrueMuE}},

};

const std::map<std::string, ana::Cut>  CUTS
{
    {"NoCut",         ana::kNoCut},
    {"Cont",          kIsOutputContained},
    {"NumuReco",      kHasMuCandTrack},
    {"NumuReco+Cont", kHasMuCandTrack && kIsOutputContained},
    {"NumuReco+Cont+Signal", kHasMuCandTrack && kIsOutputContained && (kHasTrueMu && kIsVtxContained && kHasAllContainedEnergy)}
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

  if (spectra.count("EmuResidVsTrueEmu_NumuReco+Cont") > 0)
  {
    ana::Spectrum & spec = spectra.at("EmuResidVsTrueEmu_NumuReco+Cont");
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
