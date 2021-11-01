#include <numeric>

#include "TCanvas.h"
#include "TH1.h"
#include "TH2.h"

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
    {"Emu", {"True muon energy (GeV)",           ana::Binning::Simple(50, 0, 5), kTrueMuE,
                "Muon candidate track length (cm)", ana::Binning::Simple(35, 0, 700), kMuonCandLen}},

};

const std::map<std::string, ana::Cut>  CUTS
{
    {"NoCut",         ana::kNoCut},
    {"Cont",          kIsOutputContained},
    {"NumuReco",      kHasMuCandTrack},
    {"NumuReco+Cont", kHasMuCandTrack && kIsOutputContained},
};


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
}
