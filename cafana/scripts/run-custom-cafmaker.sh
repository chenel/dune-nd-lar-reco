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

if [ $# -ne 7 ]; then
  echo "Invalid argument count: $#"
  echo "Usage: $0 <ups tarball name> <cafmaker tarball name> <dumpfile> <ghep file> <ndlar-reco h5 summary file> <seed> <full path to outfile>"
  echo "  Note: outfile path must be accessible by 'ifdh cp'"
  exit 1
fi

UPS_TARBALL="$CONDOR_DIR_INPUT/$1"
CAFMAKER_TARBALL="$CONDOR_DIR_INPUT/$2"
DUMP_FILE="$CONDOR_DIR_INPUT/$3"
GHEP_FILE="$CONDOR_DIR_INPUT/$4"
H5_FILE="$CONDOR_DIR_INPUT/$5"
SEED="$6"
OUTFILE="$7"

shift 7

echo "available transferred files:"
ls -l $CONDOR_DIR_INPUT

echo "checking your input files:"

# first check the input files
check_input_file "$DUMP_FILE" "dumpTree format input file"
check_input_file "$GHEP_FILE" "ghep format input file"
check_input_file "$H5_FILE" "ND-LAr-reco h5 summary input file"

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
. ndcaf_setup.sh || exit 1
popd > /dev/null



# now it is time to go
echo "running the CAFMaker:"
cd $_CONDOR_SCRATCH_DIR
cp -r $CAFMAKER_TARBALL/nusyst_inputs .  # this HAS to be in the cur-dir b/c of the file paths in the fcl
outfile_stub="$( basename $OUTFILE )"
time makeCAF --infile $DUMP_FILE \
             --gfile $GHEP_FILE \
             --ndlar-reco $H5_FILE  \
             --outfile $outfile_stub \
             --fhicl $CAFMAKER_TARBALL/sim_inputs/fhicl.fcl \
             --seed $SEED \
             --oa 0m \
  || exit $?

echo "now copying outfile:"
export IFDH_CP_MAXRETRIES=0
ifdh cp $outfile_stub $OUTFILE

echo "finished successfully!"
exit 0