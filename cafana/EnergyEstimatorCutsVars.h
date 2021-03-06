
#include "CAFAna/Core/Cut.h"
#include "CAFAna/Core/Var.h"

#include "StandardRecord/Proxy/SRProxy.h"
#include "duneanaobj/StandardRecord/SRNDLAr.h"

/// based on https://indico.fnal.gov/event/48610/#4-ml-reconstruction-update, p. 10
const float MIN_TRACK_LENGTH = 100.0;  // cm

/// correct for the offset subtracted by the CAFMaker...
const float OFFSET[3] = {0., 0., 0.};

/// these are from JW's explorations of simulated files.
/// they are based on the bounds used for the LAr reconstruction,
/// which extend just outside the active LAr volume,
/// but brought back in 10cm on each side
const float FID_VOL_LOW_EXTENT[3] = {-340, -150,  420};
const float FID_VOL_HIGH_EXTENT[3] = {340,  75,  900};
//(-350., 350.), (-217., 83.), (418., 914.)

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
using namespace ana;
ana::Var kTrueVtxX = SIMPLEVAR(vtx_x);
ana::Var kTrueVtxY = SIMPLEVAR(vtx_y);
ana::Var kTrueVtxZ = SIMPLEVAR(vtx_z);

ana::Var kTrueLepEndX = SIMPLEVAR(LepEndpoint.x);
ana::Var kTrueLepEndY = SIMPLEVAR(LepEndpoint.y);
ana::Var kTrueLepEndZ = SIMPLEVAR(LepEndpoint.z);

ana::Var kNTracks = SIMPLEVAR(ndlar.ntracks);

ana::Var kTrueMuE([](const caf::SRProxy * sr) -> float
                  {
                    if (std::abs(sr->LepPDG) != 13)
                      return -999.;

                    return sr->LepE;
                  });

ana::Var kTrueEnu = SIMPLEVAR(Ev);

ana::Var kTrueInel([](const caf::SRProxy * sr) -> float
                  {
                    return 1. - sr->LepE / sr->Ev;
                  });

ana::Var kTrueLepPDG = SIMPLEVAR(LepPDG);

// ------------------------------------
/// Returns (index, length) pair of the muon candidate (longest track) if there is one
std::pair<int, float> MuonCand(const caf::SRProxy * sr)
{
  static std::pair<int, float> cached{-1, -1};
  static std::tuple<int, int, int> cachedEvt{-1, -1, -1};
  if (cachedEvt == std::make_tuple(sr->run, sr->subrun, sr->event))
    return cached;

  if (sr->ndlar.ntracks < 1)
    return {-1, -1};

  std::pair<std::size_t, float> longest = {-1, -1};
  for (std::size_t idx = 0; idx < sr->ndlar.ntracks; idx++)
  {
    const caf::SRTrackProxy & tr = sr->ndlar.tracks[idx];
    float length = caf::SRVector3D(tr.end.x - tr.start.x,
                                   tr.end.y - tr.start.y,
                                   tr.end.z - tr.start.z).Mag();
    if (length > longest.second)
      longest = {idx, length};
  }
  cachedEvt = std::make_tuple(sr->run, sr->subrun, sr->event);
  cached = longest;

  return longest;
}

#define MUON_VERTEX_VAR(VARNAME, COORD) ana::Var VARNAME([](const caf::SRProxy* sr) -> float { \
  int idx; float len; \
  std::tie(idx, len) = MuonCand(sr); \
  if (idx < 0) return -9999.; \
  return sr->ndlar.tracks[idx].start.COORD; \
});

MUON_VERTEX_VAR(kMuonCandVtxX, x);
MUON_VERTEX_VAR(kMuonCandVtxY, y);
MUON_VERTEX_VAR(kMuonCandVtxZ, z);

#define MUON_END_VAR(VARNAME, COORD) ana::Var VARNAME([](const caf::SRProxy* sr) -> float { \
  int idx; float len; \
  std::tie(idx, len) = MuonCand(sr); \
  if (idx < 0) return -9999.; \
  return sr->ndlar.tracks[idx].end.COORD; \
});

MUON_END_VAR(kMuonCandEndX, x);
MUON_END_VAR(kMuonCandEndY, y);
MUON_END_VAR(kMuonCandEndZ, z);


ana::Var kMuonCandLen([](const caf::SRProxy * sr) -> float
                      {
                        if (sr->ndlar.ntracks < 1)
                          return -999.;

                        // tracks are meant to be sorted in descending order (longest first)
                        return MuonCand(sr).second;
                      });

/// sum of visible energies of all non-muon-candida.te tracks (i.e. not longest one), in GeV
ana::Var kNonMuonCandTotalTrkVisE([](const caf::SRProxy * sr) -> float
                                  {
                                    if (sr->ndlar.ntracks < 2)
                                      return 0.;

                                    // skip the muon candidate!
                                    float totalVisE = 0;
                                    for (std::size_t idx = 0; idx < sr->ndlar.ntracks; idx++)
                                    {
                                      if (static_cast<int>(idx) == MuonCand(sr).first)
                                        continue;
                                      totalVisE += sr->ndlar.tracks[idx].Evis * 1e-3; // convert to GeV
                                    }

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

ana::Var kNShowers = SIMPLEVAR(ndlar.nshowers);

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
                                 return IsContained(caf::SRVector3D{sr->vtx_x + OFFSET[0], sr->vtx_y + OFFSET[1], sr->vtx_z + OFFSET[2]},
                                                    FID_VOL_LOW_EXTENT, FID_VOL_HIGH_EXTENT);
                               });

// signal definition:
// (3) are the outgoing particles contained?
// todo: what we have here is currently a bad proxy (muon is contained).
//       needs to include hadron part too
const ana::Cut kIsOutputContained([](const caf::SRProxy * sr) -> bool
                                  {
                                    return IsContained(caf::SRVector3D{sr->LepEndpoint.x + OFFSET[0],
                                                                       sr->LepEndpoint.y + OFFSET[1],
                                                                       sr->LepEndpoint.z + OFFSET[2]},
                                                       FID_VOL_LOW_EXTENT, FID_VOL_HIGH_EXTENT);
                                  });

const ana::Cut kIsSignal = kHasTrueMu && kIsVtxContained && kIsOutputContained;

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

// energy estimators:
// use FitEmuEstimator.C and FitEnuEstimator.C to find the coeffs

/// Estimates muon energy in GeV, using longest track's range
ana::Var kRecoEmuFromTrkLen([](const caf::SRProxy * sr) -> float
{
  if (sr->ndlar.ntracks < 1)
    return -999.;

  // output of FitEmuEstimator.C
  return 0.0023 * kMuonCandLen(sr) + 0.12;  // note that kMuonCandLen returns length in cm
//  return 0.0025 * kMuonCandLen(sr);  // note that kMuonCandLen returns length in cm
});

/// Total visible energy of reco tracks & showers
ana::Var kRecoHadVisE([](const caf::SRProxy * sr) -> float
{
  std::cout << "   non-mu track visE = " << kNonMuonCandTotalTrkVisE(sr) << std::endl;
  std::cout << "   shower visE = " << kTotalShwVisE(sr) << std::endl;
  float ret = kNonMuonCandTotalTrkVisE(sr) + kTotalShwVisE(sr);
  std::cout << "  --> sum = " << ret << std::endl;

  return ret;
});

/// Estimates reco Ehad from visible hadronic energy (non-muon tracks and any showers)
ana::Var kRecoEhadFromEhadVis([](const caf::SRProxy * sr) -> float
{
  // output of FitEnuEstimator.C
  return 1.3 * kRecoHadVisE(sr) + 0.34;
});
