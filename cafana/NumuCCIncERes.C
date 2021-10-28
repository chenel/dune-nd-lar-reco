#include <numeric>

#include "TCanvas.h"
#include "TH1.h"
#include "TH2.h"

#include "CAFAna/Core/Spectrum.h"
#include "CAFAna/Core/SpectrumLoader.h"
#include "StandardRecord/Proxy/SRProxy.h"
#include "duneanaobj/StandardRecord/SRNDLAr.h"

#include "CAFAna/Core/Cut.h"
#include "CAFAna/Core/Var.h"

/// these are from JW's explorations of simulated files.
/// they are based on the bounds used for the LAr reconstruction,
/// which extend just outside the active LAr volume,
/// but brought back in 10cm on each side
const float FID_VOL_LOW_EXTENT[3] = {-360, -150,  390};
const float FID_VOL_HIGH_EXTENT[3] = {360,  150,  920};

/// based on https://indico.fnal.gov/event/48610/#4-ml-reconstruction-update, p. 10
const float MIN_TRACK_LENGTH = 100.0;  // cm

// ------------------------------------

template <typename T>
bool IsContained(const T & vec, const float * low_extent, const float * high_extent)
{
  static_assert(std::is_same_v<T, caf::SRVector3D> || std::is_same_v<T, caf::SRVector3DProxy>,
                "IsContained() only works on SRVector3D or SRVector3DProxy");
  if (vec.x < low_extent[0] || vec.x > high_extent[0])
    return false;

  if (vec.y < low_extent[1] || vec.y > high_extent[1])
    return false;

  if (vec.z < low_extent[2] || vec.z > high_extent[2])
    return false;

  return true;
}


// ------------------------------------

ana::Var kNTracks([](const caf::SRProxy * sr) -> float
{
  return sr->ndlar.ntracks;
});

ana::Var kTrueMuE([](const caf::SRProxy * sr) -> float
{
  if (std::abs(sr->LepPDG) != 13)
    return -999.;

  return sr->LepE;
});

ana::Var kTrueEnu([](const caf::SRProxy * sr) -> float
{
  return sr->Ev;
});

// ------------------------------------

ana::Var kMuonCandLen([](const caf::SRProxy * sr) -> float
{
  if (sr->ndlar.ntracks < 1)
    return -999.;

  // tracks are meant to be sorted in descending order (longest first)
  return caf::SRVector3D(sr->ndlar.tracks[0].end.x - sr->ndlar.tracks[0].start.x,
                         sr->ndlar.tracks[0].end.y - sr->ndlar.tracks[0].start.y,
                         sr->ndlar.tracks[0].end.z - sr->ndlar.tracks[0].start.z).Mag();
});

/// sum of visible energies of all non-muon-candidate tracks (i.e. not longest one), in GeV
ana::Var kNonMuonCandTotalTrkVisE([](const caf::SRProxy * sr) -> float
{
  if (sr->ndlar.ntracks < 2)
    return 0.;

  // tracks are meant to be sorted in descending order (longest first),
  // so skip the first one
  float totalVisE = 0;
  for (std::size_t idx = 1; idx < sr->ndlar.ntracks; idx++)
    totalVisE += sr->ndlar.tracks[idx].Evis * 1e-3; // convert to GeV

//  std::cout << "total non-'mu' track visE = " << totalVisE << std::endl;
  return totalVisE;
});

/// sum of visible energies of all showers in GeV
ana::Var kTotalShwVisE([](const caf::SRProxy * sr) -> float
{
  if (sr->ndlar.nshowers < 1)
    return 0.;

  float totalVisE = 0;
  for (const auto & shw : sr->ndlar.showers)
    totalVisE += shw.Evis * 1e-3; // convert to GeV

//  std::cout << "total shower visE: " << totalVisE;
  return totalVisE;
});

// ------------------------------------

// signal definition:
// (1) is there a true muon candidate in it?
const ana::Cut kHasTrueMu([](const caf::SRProxy * sr) -> bool
{
  return std::abs(sr->LepPDG) == 13;
});

// signal definition:
// (2) is the vertex contained?
const ana::Cut kIsVtxContained([](const caf::SRProxy * sr) -> bool
{
    return IsContained(caf::SRVector3D{sr->vtx_x, sr->vtx_y, sr->vtx_z}, FID_VOL_LOW_EXTENT, FID_VOL_HIGH_EXTENT);
});

// signal definition:
// (3) are the outgoing particles contained?
// todo: what we have here is currently a bad proxy (muon is contained).
//       needs to include hadron part too
const ana::Cut kIsOutputContained([](const caf::SRProxy * sr) -> bool
{
  return IsContained(sr->LepEndpoint, FID_VOL_LOW_EXTENT, FID_VOL_HIGH_EXTENT);
});

// ------------------------------------

// reco cuts:
//  (1) we have a sufficiently long track
ana::Cut kHasMuCandTrack([](const caf::SRProxy * sr) -> bool
{
  if (sr->ndlar.ntracks == 0)
    return false;

  // tracks are meant to be sorted in descending order (longest first)
  return kMuonCandLen(sr) > MIN_TRACK_LENGTH;
});

// reco cuts:
//   (2) all the energy is contained
ana::Cut kHasAllContainedEnergy([](const caf::SRProxy * sr) -> bool
{
  for (const auto & tr : sr->ndlar.tracks)
  {
    if (!IsContained(tr.start, FID_VOL_LOW_EXTENT, FID_VOL_HIGH_EXTENT)
        || !IsContained(tr.end, FID_VOL_LOW_EXTENT, FID_VOL_HIGH_EXTENT))
      return false;
  }

  // todo: seems like we need some way to measure shower extent too...
  for (const auto & shw : sr->ndlar.showers)
  {
    if (!IsContained(shw.start, FID_VOL_LOW_EXTENT, FID_VOL_HIGH_EXTENT))
      return false;
  }

  return true;
});

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
    {"NoCut",            ana::kNoCut},
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
