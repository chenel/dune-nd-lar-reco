#include "TCanvas.h"
#include "TFitResult.h"
#include "TLatex.h"
#include "TH1.h"
#include "TH2.h"
#include "TUnixSystem.h"

#include "CAFAna/Core/LoadFromFile.h"
#include "CAFAna/Core/Spectrum.h"
#include "CAFAna/Core/SpectrumLoader.h"

#include "EnergyEstimatorCutsVars.h"

const std::string HIST_LABEL = "RecoTrkLenVsTrueEmu";

// ------------------------------------------------------------------------

/// Assuming the 2D histogram is reco track length (y) vs. true muon energy (x),
/// create a TProfile of true muon energy vs. reco track length, which will be fitted later.
/// This has profile center at the *peak* of the x-bins in each row (y-bins), not the mean
TH1D ProfileHist(const TH2 * h)
{
  TH1D profile("recoEmu_profiled", "", h->GetNbinsY(), h->GetYaxis()->GetXmin(), h->GetYaxis()->GetXmax());
  profile.SetTitle(";Muon candidate track length (cm);True muon energy (GeV)");
  // we walk up the y-axis bins...
  for (int y = 1; y <= h->GetNbinsY(); y++)
  {
    TH1D * proj = h->ProjectionX(Form("proj_bin%d", y), y, y);
    profile.SetBinContent(y, proj->GetBinCenter(proj->GetMaximumBin()));

    // figure out the smallest interval that contains both ~68.2% of the distribution and the max.
    // we scan from the first bin up to the max bin
    int maxBin = proj->GetMaximumBin();
    TH1 * cumul = proj->GetCumulative();
    std::cout << "For y bin " << y << "< cumulative distribution bins:" << std::endl;
    for (int i = 1; i <= cumul->GetNbinsX();  i++)
      std::cout << " " << cumul->GetBinContent(i) ;
    std::cout << std::endl;
    if (cumul->GetBinCenter(cumul->GetNbinsX()) > 0)
      cumul->Scale(1. / cumul->GetBinContent(cumul->GetNbinsX()));
    std::cout << "after rescaling:" << std::endl;
    std::cout << "For y bin " << y << "< cumulative distribution bins:" << std::endl;
    for (int i = 1; i <= cumul->GetNbinsX();  i++)
      std::cout << " " << cumul->GetBinContent(i) ;
    std::cout << std::endl;

    std::vector<double> intervals;
    for (int xbin = 1; xbin < maxBin; xbin++)
    {
      for (int testBin = xbin; testBin < proj->GetNbinsX(); testBin++)
      {
        std::cout << "for y bin " << y << ", interval from xbin=" << xbin << "to xbin=" << testBin << " contains  " << cumul->GetBinContent(testBin) - cumul->GetBinContent(xbin) << " of prob" << std::endl;
        if (cumul->GetBinContent(testBin) - cumul->GetBinContent(xbin) > 0.682)
        {
          // we only care about the interval if it includes the max bin...
          if (testBin >= maxBin)
          {
            std::cout << " interval size = " << cumul->GetBinCenter(testBin) - cumul->GetBinCenter(xbin) << std::endl;
            intervals.emplace_back(cumul->GetBinCenter(testBin) - cumul->GetBinCenter(xbin));
          }
          break;
        }
      }
    }

    // now find the smallest interval.
    // use a custom comparator because we want to ignore
    // intervals that didn't include the maximum,
    // which have interval size < 0
    std::cout << "considering intervals:" << std::endl;
    for (const auto & interval : intervals)
      std::cout << "  " << interval;
    std::cout << std::endl;
    double smallest = *std::min_element(intervals.begin(), intervals.end());

    assert (smallest >= 0 && Form("smallest 68.2%% interval is negative??: %f", smallest));

    // this apes the error on the mean (sigma / sqrt(N)).
    // sadly it symmetrizes what should be an asymmetric error,
    // but I'm not interested in TGraphAsymmErrors today
    std::cout << "for ybin=" << y << ", smallest 68.2% interval is " << smallest << std::endl;
//    profile.SetBinError(y, smallest / proj->GetEntries());
    profile.SetBinError(y, smallest);
  }

  return profile;
}

// ------------------------------------------------------------------------
/// Driver.
class FitDriver
{
  public:
    FitDriver(const std::string & outdir, const TH2 * h)
      : fOutDir(outdir), fHist(h)
    {}

    void DoFit();

    const TH2 * Hist() const { return fHist; }
    const std::string & OutDir() const { return fOutDir; }

    void SaveCanvasImg(const TCanvas & c, const std::string & stub) const
    {
      for (const auto & ext : fImgExt)
        c.SaveAs( (OutDir() + "/" + stub + "." + ext).c_str() );
    }

    void SetImgExt(const std::vector<std::string> & ext) { fImgExt = ext; };

  private:
    std::string fOutDir;
    const TH2 * fHist;

    std::vector<std::string> fImgExt = { "png", "pdf", "root" };
};

void FitDriver::DoFit()
{
  TH1D prof = ProfileHist(this->Hist());


  TCanvas c;
  prof.SetMarkerSize(10);
  prof.Draw("pe");
  TFitResultPtr fit = prof.Fit("pol1", "s");
//  fit->Draw("same");
  TLatex text(25, 3, Form("E_{#mu}/GeV = %.2g L_{trk}/cm + %.2f", fit->GetParams()[1], fit->GetParams()[0]));
  text.Draw();
  this->SaveCanvasImg(c, "MuonTrkE_prof_TrueMuonE");
}


// ------------------------------------------------------------------------
// ------------------------------------------------------------------------

// only 2 arguments: hist file already exists
void FitEmuEstimator(const std::string & histFilePath, const std::string & outDir)
{
  auto specPtr = ana::LoadFromFile<ana::Spectrum>(histFilePath, HIST_LABEL);
  if (!specPtr)
  {
    std::cerr << "Unable to load hist '" << HIST_LABEL << "' from file '" << histFilePath << "'.  Check your file!" << std::endl;
    exit(1);
  }

  auto th2 = specPtr->ToTH2(specPtr->POT());
  if (!th2)
  {
    std::cerr << "Could not convert Spectrum to TH2!  Abort" << std::endl;
    exit(1);
  }

  FitDriver f(outDir, th2);
  f.DoFit();
}

// 3 arguments: requesting to make the hist file first, then fit
void FitEmuEstimator(const std::string & inputCAF, const std::string & histFilePath, const std::string & outDir)
{
  ana::SpectrumLoader loader(inputCAF);
  ana::Spectrum spec_RecoTrkLen_vs_TrueEmu(loader,
                                           ana::HistAxis("True muon energy (GeV)", ana::Binning::Simple(50, 0, 5), kTrueMuE),
                                           ana::HistAxis("Muon candidate track length (cm)", ana::Binning::Simple(35, 0, 700), kMuonCandLen),
                                           kHasAllContainedEnergy);

  loader.Go();

  TFile outf(histFilePath.c_str(), "recreate");
  ana::SaveTo(spec_RecoTrkLen_vs_TrueEmu, &outf, HIST_LABEL);

  TH2 * h = spec_RecoTrkLen_vs_TrueEmu.ToTH2(spec_RecoTrkLen_vs_TrueEmu.POT());
  FitDriver f(outDir, h);
  f.DoFit();
}
