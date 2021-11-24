#!/bin/bash

check_input_file()
{
  if [ -f "$1" ] || [ -d "$1" ]; then
    return
  else
     echo "Couldn't find $2 at expected location: $1"
     echo "Contents of \$CONDOR_DIR_INPUT='$CONDOR_DIR_INPUT' are:"
     ls -lh $CONDOR_DIR_INPUT
     echo "Abort."
     exit 1
  fi
}

if [ $# -ne 5 ]; then
  echo "Invalid argument count: $#"
  echo "Usage: $0 <ups tarball name> <cafmaker tarball name> <edep-sim file> <seed> <full path to outfile>"
  echo "  Note: outfile path must be accessible by 'ifdh cp'"
  exit 1
fi

UPS_TARBALL="$CONDOR_DIR_INPUT/$1"
CAFMAKER_TARBALL="$CONDOR_DIR_INPUT/$2"
EDEP_FILE="$CONDOR_DIR_INPUT/$3"
SEED="$4"
OUTFILE="$5"

shift 5

echo "available transferred files:"
ls -l $CONDOR_DIR_INPUT

echo "checking your input files:"

# first check the input files
check_input_file "$EDEP_FILE" "dumpTree format input file"


# now unroll the custom UPS tarball & set it up
#check_input_file "$UPS_TARBALL" "UPS tarball"
#pushd $_CONDOR_SCRATCH_DIR > /dev/null
#mkdir ups
#cd ups
#
#tar -xvjf $UPS_TARBALL || exit 1
#export PRODUCTS="$PWD"
#popd > /dev/null

# turns out that jobsub is automatically unrolling the tarballs for me (!)
export PRODUCTS="$PRODUCTS:`readlink -f $UPS_TARBALL`"


# finally, unroll the custom CAFMaker tarball & set it up
#check_input_file "$CAFMAKER_TARBALL" "CAFMaker tarball"
#pushd $_CONDOR_SCRATCH_DIR > /dev/null
#mkdir cafmaker
#cd cafmaker
#tar -xvjf $CAFMAKER_TARBALL || exit 1

pushd `readlink -f $CAFMAKER_TARBALL` > /dev/null
# setup stuff cribbed from run_everything.sh
source /cvmfs/dune.opensciencegrid.org/products/dune/setup_dune.sh

export GNUMIXML="$PWD/sim_inputs/GNuMIFlux.xml"
export GXMLPATH=${PWD}:${GXMLPATH}
setup ifdhc || exit 1
setup dk2nugenie   v01_06_01f -q debug:e15 || exit 1
setup genie_xsec   v2_12_10   -q DefaultPlusValenciaMEC || exit 1
setup genie_phyopt v2_12_10   -q dkcharmtau || exit 1
setup geant4 v4_10_3_p01b -q e15:prof || exit 1
export LD_LIBRARY_PATH=${PWD}/nusystematics/build/Linux/lib:${LD_LIBRARY_PATH}
export LD_LIBRARY_PATH=${PWD}/nusystematics/build/nusystematics/artless:${LD_LIBRARY_PATH}
export FHICL_FILE_PATH=${PWD}/nusystematics/nusystematics/fcl:${FHICL_FILE_PATH}
export PYTHONPATH=${PWD}/DUNE_ND_GeoEff/lib/:${PYTHONPATH}
export LD_LIBRARY_PATH=${PWD}/DUNE_ND_GeoEff/lib:${LD_LIBRARY_PATH}
popd > /dev/null

echo "dumping tree for file: $EDEP_FILE"
cd $_CONDOR_SCRATCH_DIR
cp -r $CAFMAKER_TARBALL/dumpTree.py . || exit $?
outfile_stub="$( basename $OUTFILE )"
python dumpTree.py --infile $EDEP_FILE --outfile $outfile_stub --seed $SEED  || exit $?

echo "now copying outfile:"
export IFDH_CP_MAXRETRIES=0
ifdh cp $outfile_stub $OUTFILE

echo "finished successfully!"
exit 0