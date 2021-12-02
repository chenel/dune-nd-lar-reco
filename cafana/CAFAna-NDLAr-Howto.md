# ND-LAr-reco-to-CAFAna: Howto
Dec. 2, 2021
Jeremy Wolcott <jwolcott@fnal.gov>

## 0. Housekeeping
* These notes go with a live session with a Zoom recording.  See [the Indico page](https://indico.fnal.gov/event/52169/).
* The instructions here are only for folks who are interested in getting their hands dirty!  The CAF analysis for the CAFs I've made is work in progress, and stuff is sure to be broken. I've tried to compose instructions that will work to get you started, but if something is broken, you're agreeing to work on fixing it yourself, not just ask me for help. :)
* The CAFs used for the demonstrations detailed below are currently in my user area:
    ```
    /dune/data/users/jwolcott/nd-lar-reco/caf/geom-20210623/*.root
    ```
    There are only 22 of them at the moment because there is a relatively high-frequency processing error with the ND-LAr reco itself that I have not completely untangled.
    Once I do, I'll produce the CAFs for the remaining ~80 files.
    
    These CAFs correspond to the single-neutrino simulation using the ND-LAr+support structure only (no spectrometer or interactions in the detector hall walls etc.) produced by Chris Marshall in June 2021.
    The source files (edep-sim, etc.) are in
    ```
    /pnfs/dune/persistent/users/marshalc/LArTMSProductionJun23withLArCV/
    ```
    The enumeration is the same.
    (Please note, however, that I found I needed to regenerate the "dump" files used as input to the CAFMaker because Chris's original files had a coordinate offset baked into them.  My corresponding "dump" files are in `/dune/data/users/jwolcott/nd-lar-reco/dump`.) 


## 1. Basic orientation

### Analysis basics

The DUNE Long-baseline (LBL) group uses a package called [CAFAna](https://github.com/DUNE/lblpwgtools/) to compute oscillation sensitivities.
In a nutshell, CAFAna does the following things:
* Wraps **loops over analysis ntuples** in a user-friendly way
* Provides a user-friendly interface for **aggregating histograms** (and many more complex structures built from histograms) during the aforementioned loops
* Offers extensive tools for **fitting models** to reference data via minimizers or MCMC sampling

Though the DUNE software ecosystem is currently extremely fragmented, CAFAna is (at the moment anyway) the tool that links everything together.
Basically, the workflow for sensitivity plots is something like:
* CAFs (Common Analysis Files)---analysis ntuples---are made from relevant inputs for both FD and ND samples using [the _art_ module](https://cdcvs.fnal.gov/redmine/projects/dunetpc/repository/entry/dune/CAFMaker/) (FD) or standalone [`ND_CAFMaker`](https://github.com/DUNE/ND_CAFMaker/)
* CAFAna `Prediction`s (objects consisting of reference spectra and the machinery necessary to generate spectra at arbitrary oscillation parameters and systematic uncertainty pulls) are made by running relevant CAFAna scripts over the CAFs
* Fake data spectra are generated using those `Prediction`s at agreed-upon oscillation parameters
* Fits are made to these fake data using the `Prediction`s and CAFAna fitting framework

For the purposes of this Howto, we will only concern ourselves with the first two items, which comprise the portion of the workflow relevant for the ND-LAr at present.

### Why this Howto exists

There is existing documentation for how to run the `ND_CAFMaker` (see its [`README.md`](https://github.com/DUNE/ND_CAFMaker/blob/master/README.md)) and `CAFAna` (its [`README.md`](https://github.com/DUNE/lblpwgtools/blob/master/README.md)).
However, numerous changes needed to be made to the ntuple branches stored in the CAFs in order to include ND-LAr reco information in them, and further changes are imminent from TMS reconstruction and track matching.
These have not been fully validated yet, and so they have not been merged in to the official versions.
Until that happens, users will need to build their own versions to read the CAFs produced with ND-LAr reco branches in them.

## 2. Software basics

### Packages & UPS

Fermilab's computing experts use a software packaging system called "UPS" for distributing multiple versions of packages that run on various operating systems, are built with different compilers, have different options, etc.
Though UPS is slated to be replaced with a different system called Spack in the somewhat near future, it's the machinery in place at the moment.
Some of the software needed to run CAFAna with the new CAF structure is designed to be delivered as UPS packages.

UPS has extensive documentation on [the Redmine wiki](https://cdcvs.fnal.gov/redmine/projects/ups/wiki/Getting_Started_Using_UPS) (you'll need Fermilab Services credentials to log in to view these pages).
Only the basics will be mentioned here.

#### Package identification

UPS packages are defined by four things:
* Name
* Version
* "Flavor"
* Qualifiers

For instance, a recent version of ROOT (version 6.22.08), built for Scientific Linux 7, using GCC 9.3.0 in C++17 mode and Python 3.9.2, with debugging symbols in the library would be identified as
```
"root" "v6_22_08d" "Linux64bit+3.10-2.17" "debug:e20:p392"
```
(Note that there can be multiple 'qualifiers' in the last item, and they're separated by colons.)

#### Finding and setting up packages

To enable the default UPS repositories and turn on its functionality, one uses the setup script from CVMFS:

```bash
source /cvmfs/dune.opensciencegrid.org/products/dune/setup_dune.sh
```

Then one can list available packages.  For instance, to see the variants of the package for ROOT 6.22.08, one might do

```bash
$ ups list -aK+ root v6_22_08b
"root" "v6_22_08b" "Linux64bit+3.10-2.17" "debug:e20:p383b" "" 
"root" "v6_22_08b" "Linux64bit+3.10-2.17" "c7:debug:p383b" "" 
"root" "v6_22_08b" "Linux64bit+3.10-2.17" "e20:p383b:prof" "" 
"root" "v6_22_08b" "Linux64bit+3.10-2.17" "c7:p383b:prof" "" 
```

Then to activate a particular version, one uses the `setup` command:
```bash
setup "root" "v6_22_08d" -f "Linux64bit+3.10-2.17" -q "debug:e20:p392"
```
Note that generally the flavor can be autodetected by UPS, and the preceding command could just as well have been
```bash
setup "root" "v6_22_08d" -q "debug:e20:p392"
```

### What is a CAF?  (Where ~~does babby~~ do CAFs come from?)

CAFs are ROOT ntuples.
As mentioned above, they are produced by "CAFMaker" programs---one each for the ND and FD.
In this Howto we'll only bother with the output of the ND CAFMaker, which was originally used to produce ntuples with truth-based "pseudo-reconstruction".
These were the ND CAFs the results in the FD TDR were based upon.

How to run the CAFMaker(s) is outside the scope of this Howto, though I might write another one on that subject in the future. :)

#### The `StandardRecord`
All CAFs contain a ROOT `TTree` called `cafTree` which contains one custom `StandardRecord` object corresponding to each reconstructed event.
(They also may contain a number of other `TTree`s, but we don't need to worry about them for the purposes of this Howto.)
The `StandardRecord` class is contained in the `duneanaobj` UPS product, whose source code is in the DUNE GitHub area ([DUNE/duneanaobj](https://github.com/DUNE/duneanaobj/tree/master/duneanaobj)), and it currently has a single UPS product version available (but built for a handful of architectures and against GENIE v2 or v3):
```bash
$ ups list -aK+ duneanaobj
"duneanaobj" "v01_00_00" "Linux64bit+3.10-2.17" "c2:debug:gv2" "" 
"duneanaobj" "v01_00_00" "Linux64bit+3.10-2.17" "c7:debug:gv3" "" 
"duneanaobj" "v01_00_00" "Linux64bit+3.10-2.17" "e17:gv2:prof" "" 
"duneanaobj" "v01_00_00" "Linux64bit+3.10-2.17" "debug:e20:gv3" "" 
"duneanaobj" "v01_00_00" "Linux64bit+3.10-2.17" "e20:gv3:prof" "" 
"duneanaobj" "v01_00_00" "Linux64bit+3.10-2.17" "c2:gv2:prof" "" 
"duneanaobj" "v01_00_00" "Linux64bit+3.10-2.17" "c7:gv3:prof" "" 
"duneanaobj" "v01_00_00" "Linux64bit+3.10-2.17" "debug:e17:gv2" "" 
```

The exact format of the `StandardRecord` is currently in flux, however, and the relevant changes are precisely the reason why this Howto even exists.
* Originally, the DUNE `StandardRecord` format was "flat": one branch at top level for every piece of information.  E.g.:
```c++
class StandardRecord
{
  ... 
  int meta_run;
  int meta_subrun;
  double pot;
  ...
}
```
* In other deployments of CAFAna for other experiments (particularly NOvA and SBN), experience has shown that having a more "structured" layout, where information is nested in sub-objects, is much more maintainable.  E.g.:
```c++
class StandardRecord
{
  ... 
  SRHeader         hdr; 
  SRSpill          spill;
  SRSlice          slc;
}
```
Each of these `SR` types then contains fields that can be drilled down into; they might contain further `SR` types; and so on.

#### Updates to `StandardRecord` for ND-LAr reco
As the FD CAFMaker is being prepared for a move to a more structured format (see branch [`feature/cafs`](https://cdcvs.fnal.gov/redmine/projects/dunetpc/repository/entry/dune/CAFMaker?utf8=%E2%9C%93&rev=feature%2Fcafs)), I decided to introduce the ND-LAr reco variables in a similar "structured" format.

**The relevant branch for this work in `duneanaobj` is called [`feature/add_nd_vars`](https://github.com/DUNE/duneanaobj/tree/feature/add_nd_vars).**

Besides restructuring the other branches to match what's in the FD CAFMaker, this branch notably adds a new object called `ndlar`, which currently has fields for reconstructed tracks and showers.

#### Browsing

Even though some of the branches of the `StandardRecord` are complex objects that require the `dunanaobj` library to fully deserialize, most of the structure is simple enough that you can just look at them in a ROOT `TBrowser`.
(In the live session I'll demonstrate this briefly.)
You will note that you don't actually see `StandardRecord` anywhere; this type is (mostly) automatically unrolled by the ROOT Browser and you just see its contents.

### Using CAFAna

After installation, CAFAna is typically run by writing ROOT macros that can either be executed using its driver program, `cafe`, or compiled against the CAFAna library into executables (as was done for many of the FD TDR-era macros).
More on that below.

## 3. Building your own versions 

To use the modified `StandardRecord` with ND-LAr variables in it---which you'll need to do in order to read any of the CAFs I've made---you'll need to build some of the software yourself.

### 'Local' UPS 

The most difficult part of this exercise is setting up a UPS repository containing a version of `duneanaobj` with the changes I noted above.

I typically do this in my `/dune/app/users/jwolcott` area, where there is a subdirectory called `ups`.
To make your own, simply create such a directory:
```bash
export CUSTOM_UPS="/dune/app/users/$USER/ups"
mkdir $CUSTOM_UPS
```

For UPS to recognize it properly, it needs a special hidden subdirectory called `.upsfiles` in it.
You can just copy the `.upsfiles` from the main CVMFS UPS area:
```bash
cp -r /cvmfs/dune.opensciencegrid.org/products/dune/.upsfiles $CUSTOM_UPS
```

Then, to tell UPS that this directory is a place it should search for UPS products, you need to add it to the `$PRODUCTS` environment variable:
```bash
export PRODUCTS="$PRODUCTS:$CUSTOM_UPS"
```

**In the future, any time you want to work with this 'custom' UPS area (including any time you want to read my CAFs, until this version of `duneanaobj` gets deployed officially), you'll need to do this setup!**
```bash
export CUSTOM_UPS="/dune/app/users/$USER/ups"
export PRODUCTS="$PRODUCTS:$CUSTOM_UPS"
```
(You may want to store this in a script somewhere that you can easily source.)

### Building your own package for `duneanaobj` 

Right now your 'custom' UPS area is empty.
You need to install a package of the updated `duneanaobj` in it for it to be useful.

* First, you'll want to **check out the source code**.  I usually work in `/dune/app/users/jwolcott/dunesoft/duneanaobj`, but you can modify to suit your taste:
    ```bash
    DUNEANAOBJ_SRC=/dune/app/users/jwolcott/dunesoft/duneanaobj
    git clone https://github.com/DUNE/duneanaobj.git $DUNEANAOBJ_SRC
    cd $DUNEANAOBJ_SRC
    git checkout feature/add_nd_vars
    ```
    
    At the moment this branch is configured to produce a UPS product with version `v01_01_00`, which is the ostensible next version of a public product.
    It may be advisable for you to set your custom version to something more recognizable as home-brewed---perhaps `testing`.
    To do that, edit the file `ups/product_deps` and find the line near the top beginning with `parent`.
    Swap out `v01_01_00` with whatever you've picked.
    (I'll use `testing` for the rest of this Howto.)

* Next, **set up the environment**:

    `duneanaobj` uses the scripting developed by the Fermilab SciSoft group (who maintain an extensive collection of UPS packages) to build.
    This means that it builds using CMake.
    CMake builds are (by default anyway) "out-of-source"---i.e., they happen in a disposable directory that is not within the source code.
    I usually build in a directory `/dune/app/users/jwolcott/duneanaobj-build`, but again, you can change it to suit your preferences.
    ```bash
    DUNEANAOBJ_BUILD=/dune/app/users/jwolcott/duneanaobj-build
    mkdir $DUNEANAOBJ_BUILD
    cd $DUNEANAOBJ_BUILD
    ```

* **Get ready to build**:
    ```bash
    . $DUNEANAOBJ_SRC/ups/setup_for_development -d e20:gv3
    ```
    The arguments here specify that I want a debug build (`-d`) and the qualifiers for GCC 9.3.0 in C++17 mode (`e20`) and GENIE version 3 (`gv3`).
    This should spit out a lot of output, but critically, the lines after `check this block for errors:` should not have anything in them:
    ```bash
    ----------- check this block for errors -----------------------
    ----------------------------------------------------------------
    ```
    If so, then you're ready to run the build process.

* Now **build**.

    The output from the previous step suggests the build command to use.
    It's important to specify the correct directory for the result to be installed in here, using the `$CUSTOM_UPS`  we defined earlier:
    ```bash
    buildtool -I "$CUSTOM_UPS" -bti
    ```
    
    You *may* encounter a build failure with an error like
    ```
    INFO: class 'caf::StandardRecord' has a different checksum for ClassVersion 16. Incrementing ClassVersion t
    o 17 and assigning it to checksum 3481006127
    WARNING: classes_def.xml files have been updated: rebuild dictionaries.
    ```
    If so, this isn't a catastrophe.
    Simply do
    ```bash
    make
    ```
    and the build should resume.  Assuming it's successful, then you can
    ```bash
    make install
    ```
    which will deploy the fully-constructed package to `$CUSTOM_UPS`.

At this point, you should be able to see it in the package list:
```bash
# is your $CUSTOM_UPS at the end?
$ echo $PRODUCTS
/cvmfs/dune.opensciencegrid.org/products/dune:/cvmfs/larsoft.opensciencegrid.org/products:/cvmfs/fermilab.o
pensciencegrid.org/products/common/db/:/dune/app/users/jwolcott/ups

$ ups list -aK+ duneanaobj
$ ups list -aK+ duneanaobj
"duneanaobj" "v01_00_00" "Linux64bit+3.10-2.17" "c2:debug:gv2" "" 
"duneanaobj" "v01_00_00" "Linux64bit+3.10-2.17" "c7:debug:gv3" "" 
"duneanaobj" "v01_00_00" "Linux64bit+3.10-2.17" "e17:gv2:prof" "" 
"duneanaobj" "v01_00_00" "Linux64bit+3.10-2.17" "debug:e20:gv3" "" 
"duneanaobj" "v01_00_00" "Linux64bit+3.10-2.17" "e20:gv3:prof" "" 
"duneanaobj" "v01_00_00" "Linux64bit+3.10-2.17" "c2:gv2:prof" "" 
"duneanaobj" "v01_00_00" "Linux64bit+3.10-2.17" "c7:gv3:prof" "" 
"duneanaobj" "v01_00_00" "Linux64bit+3.10-2.17" "debug:e17:gv2" "" 
"duneanaobj" "testing" "Linux64bit+3.10-2.17" "debug:e20:gv3" ""   # <-- this is the new one.  it worked! 
```

You can set up this newly installed, custom version of `duneanaobj` the same was as `root` in "Finding and setting up packages" under section 1 above:
```bash
setup duneanaobj testing -q debug:e20:gv3
```
If it worked, you should now have some `$DUNEANAOBJ_*` environment variables that point to the install location in `$CUSTOM_UPS`:
```bash
$ printenv | grep DUNEANAOBJ
DUNEANAOBJ_FQ_DIR=/dune/app/users/jwolcott/ups/duneanaobj/testing/slf7.x86_64.e20.gv3.prof
DUNEANAOBJ_INC=/dune/app/users/jwolcott/ups/duneanaobj/testing/include
DUNEANAOBJ_VERSION=v01_00_00
DUNEANAOBJ_DIR=/dune/app/users/jwolcott/ups/duneanaobj/testing
SETUP_DUNEANAOBJ=duneanaobj v01_01_00 -f Linux64bit+3.10-2.17 -z /dune/app/users/jwolcott/ups -q e20:gv3:prof
DUNEANAOBJ_LIB=/dune/app/users/jwolcott/ups/duneanaobj/testing/slf7.x86_64.e20.gv3.prof/lib
```

You may find you are missing some of these variables, e.g.:
```bash
$ printenv | grep DUNEANAOBJ
DUNEANAOBJ_DIR=/dune/app/users/jwolcott/ups/duneanaobj/testing
SETUP_DUNEANAOBJ=duneanaobj testing -f Linux64bit+3.10-2.17 -z /dune/app/users/jwolcott/ups -q debug:e20:gv3
```
There is an issue with the `setup_for_development` script above that I haven't (yet) worked out which sometimes causes the special file in the `ups` directory that tells UPS how to do the `setup` to be truncated.
In this case, you can simply copy-paste a corrected version that I have in my working area into your UPS installation:
```bash
cp /dune/app/users/jwolcott/dunesoft/duneanaobj/ups/duneanaobj.table $CUSTOM_UPS/duneanaobj/testing/ups/
```
Then, you can
```bash
unsetup duneanaobj   # may complain about "Action failed on parsing ..." -- don't worry about that
```
and try the `setup` and `printenv` commands above again.

### Building `CAFAna`

CAFAna's build instructions (in its [README.md](https://github.com/DUNE/lblpwgtools/blob/master/README.md)) are quite good.
However, in order to read the CAFs I've made, there are a few modifications necessary.

* **Check out branch `feature/ndlar-cafs-wip`**

    Again change the checkout directory to suit your preferences...
    ```bash
    CAFANA_SRC_DIR=/dune/app/users/jwolcott/dunesoft/lblpwgtools
    git clone https://github.com/DUNE/lblpwgtools.git $CAFANA_SRC_DIR
    
    cd $CAFANA_SRC_DIR
    git checkout feature/ndlar-cafs-wip
    ```

* Tell CAFAna to **pick up your custom version of `duneanaobj`**:

    Edit the file `CAFAna/cmake/ups_env_setup.sh`.
    Find the line that begins `setup duneanaobj`, and change it to match the UPS declaration of your package (in my case above, the version is `testing` and the qualifiers are `debug:e20:gv3`).
    
    **Be sure that your custom UPS area is set up at this point.** (Check the contents of `$PRODUCTS` as above.)

* Now **build**.

    Again the `lblpwgtools` package builds with CMake, though in this case, there's a helper script that manages the build directory for you---all you need to do is specify where the eventual installed copy should go.
    
    ```bash
    CAFANA_INSTALL=/dune/app/users/jwolcott/cafana-ndlar
    
    cd $CAFANA_SRC_DIR/CAFAna
    ./standalone_configure_and_build.sh -u -j 3 -I $CAFANA_INSTALL 
    ```

To use it (I recommend starting a fresh shell, not continuing from the one you built it in), you'll just need to set up your `$PRODUCTS` and then use the `CAFAnaEnv.sh` script in the install directory:
```bash
# from fresh shell
# modify to suit your actual installation areas
export CUSTOM_UPS="/dune/app/users/$USER/ups"
export CAFANA_INSTALL=/dune/app/users/jwolcott/cafana-ndlar

export PRODUCTS="$PRODUCTS:$CUSTOM_UPS"
source $CAFANA_INSTALL/CAFAnaEnv.sh
```

## 4. Example CAFAna scripts

At this point, you're set up to run CAFAna with the CAFs!

If you are new to CAFAna,  I first recommend having a look at the CAFAna 'tutorial' macros, which live in [`CAFAna/tute`](https://github.com/DUNE/lblpwgtools/tree/master/CAFAna/tute).
You can run them, e.g.:
```bash
# uses $CAFANA_SRC_DIR from above.  set that if you haven't already in your shell!
cafe -bq $CAFANA_SRC_DIR/CAFAna/tute/demo0.C
```
You will need to edit them to change the paths they use to correspond to the CAFs I mentioned at the top of this Howto.
You'll probably also need to edit some of them (`demo0.C`, for instance) to actually write out images somewhere instead of expecting you to have an interactive session.

Once you have the basics down, if you are in need of inspiration for what sort of things you can do, you can have a look at the macros I used to generate the plots included in the `Enu estimator physics slides` attached to the Indico event (see link at top of this doc).
The code is in my personal GitHub repository in [the same directory as this Howto](https://github.com/chenel/dune-personal/tree/master/nd/nd-lar).

If you like, you can clone it:
```bash
cd /where/you/want/it
git clone https://github.com/chenel/dune-personal.git jwolcott-dune-personal
cd jwolcott-dune-personal/nd/nd-lar
```
(There's lots of other various DUNE-related stuff there, which you can peruse at your leisure, but this directory is the only one with anything relevant.)

For instance, you can run the macro that makes most of the plots with:

```bash
# change this to suit your preferred dir
PLOT_DIR="/dune/data/users/jwolcott/nd-lar-reco/plots/geom-20210623"

cafe -bq -nr NumuCCIncPlots.C '/dune/data/users/jwolcott/nd-lar-reco/caf/geom-20210623/*.root' $PLOT_DIR
```

## 5. Known issues

There are a few known issues with the CAFs and the software stack as they stand.  Please feel free to attempt fixing them if you like!

1. There are `NaN`s in some of the branches.  This will (unfortunately) cause CAFAna to spew massive amounts of messages like
```
SRProxy: Warning: eRecoOther = nan in entry <entry> of <file>
```
at you.  They're harmless (unless you're trying to use the variable in question!), though fairly annoying.
Fixing this will require some edits to the `ND_CAFMaker`, which was not covered in this Howto.

2. None of the "xsec weight" branches actually work right (though some of them are filled).  There's currently a hack in the `feature/ndlar-cafs-wip` branch of CAFAna which simply bypasses the normal use of the CV weights.  I'd love for somebody who has (or wants to develop) the relevant expertise to figure out what's wrong here.

3. The issue noted above with the UPS table generated by the build scripts in `duneanaobj` seems like it should be a simple one.  If someone who has more experience fiddling with UPS packages wants to take a crack at it, I'd be very appreciative.  

4. At the moment there is no integration of TMS (or ND-GAr) reco or track matching.  I am aware of effort within the TMS group to do this, but it hasn't converged yet.

(to be continued ...) 
