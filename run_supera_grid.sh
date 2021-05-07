#!/bin/bash

###########################################
#
#  run_supera_grid.sh:
#
#    Run the "Supera" modules from larcv2 on a given edep-sim output file.
#
#    Original author: J. Wolcott <jwolcott@fnal.gov>
#               Date: May 2021
#
###########################################

SAM_PROJECT_NAME="$1"
OUT_DIR="$2"
shift 2

# ------------------------------------
# set up stuff from CVMFS

export qual="e15:prof"
source /cvmfs/dune.opensciencegrid.org/products/dune/setup_dune.sh
setup ifdhc
setup gcc v6_4_0
setup root v6_12_06a -q $qual
setup geant4 v4_10_3_p01b -q $qual
setup clhep v2_3_4_5c -q $qual
setup python v2_7_14b
setup python_future_six_request  v1_3 -q python2.7-ucs2
#setup numpy v1_14_3 -q e17:p2714b:prof


# extra stuff required by edep-sim
G4_cmake_file=`find ${GEANT4_FQ_DIR}/lib64 -name 'Geant4Config.cmake'`
export Geant4_DIR=`dirname $G4_cmake_file`
export PATH=$PATH:$GEANT4_FQ_DIR/bin

echo "Here are the files in my current workdir:"
ls -1

# ------------------------
# set up the custom products we passed in.
# they should be untarred into our working area...

# this has to be done in the edep-sim dir or edep-sim will complain
pushd edep-sim || exit 1
. setup.sh
popd

pushd larcv2 || exit 1
. configure.sh
popd


# -------------------------
# get our input file

projurl=`ifdh findProject $SAM_PROJECT_NAME ${SAM_STATION:-$EXPERIMENT}`
consumer_id=$(IFDH_DEBUG= ifdh establishProcess "$projurl" "larcv" "0.0" "`hostname`" "$GRID_USER" "python" "larcv" "")

echo "SAM project url: $projurl"
echo "SAM consumer id: $consumer_id"

uri=`IFDH_DEBUG= ifdh getNextFile $projurl $consumer_id | tail -1`
if [ -z "${uri}" ]; then
	echo "Couldn't get URI for next file... abort."
	exit 1
fi
echo "Next file to retrieve: $uri"

IFDH_DEBUG= ifdh fetchInput "$uri" > fetch.log 2>&1
if [ "$?" -ne 0 ]; then
	echo "Failed to file fetch from $uri"
	cat fetch.log
	rm fetch.log
	exit 1
fi

fname=`tail -1 fetch.log`
ifdh updateFileStatus $projurl  $consumer_id $fname transferred
echo "Retrieved file: $fname"

# -------------------------
# actually run larcv2 on it

supera_dir=${LARCV_BASEDIR}/larcv/app/Supera
if time python ${supera_dir}/run_supera.py ${supera_dir}/supera-ndlar.newgeom.cfg $fname; then
	ifdh updateFileStatus $projurl  $consumer_id $fname consumed
	ifdh setStatus "$projurl" "$consumer_id"  completed
else
	ifdh updateFileStatus $projurl  $consumer_id $fname skipped
	ifdh setStatus "$projurl" "$consumer_id"  bad
fi

#output file is always called supera.root
outfname="`basename ${fname/.edep.root/.larcv.root}`"
mv supera.root $outfname

# -------------------------
# copy back output, clean up

ifdh addOutputFile $outfname
ifdh copyBackOutput "$OUT_DIR/"

ifdh endProcess "$projurl" "$consumer_id"

ifdh cleanup -x
