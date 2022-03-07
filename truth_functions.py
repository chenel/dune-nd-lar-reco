"""
  truth_functions.py : Helper functions for working with truth info.

  Original author:  J. Wolcott <jwolcott@fnal.gov>
             Date:  Mar. 2022
"""

import numpy

import utility_functions


def true_muon_voxidxs_by_cluster(vals):
	"""
	Build a map from all true muons in the spill to the indices of the voxels in the data.

	:return:  dict of arrays of indices in vals["input_data"] corresponding to each muon, keyed by true muon cluster label
	"""

	# vals["cluster_label"][:, 8] is PDG code
	key = "true_muon_vox_by_cluster"

	if key not in vals:
		# Both mu+ and mu- SHOULD get value "2" in column 9.
		# Unfortunately this info is currently broken for reasons not yet known.
		# (It only seems to be correctly filled for the first muon encountered in the spill.)
		# So we have to bootstrap from the full particle list. :(
		part_pdg_group = numpy.array([(p.pdg_code(), p.group_id()) for p in vals["particles_raw"]])
		true_muon_groups = numpy.unique(part_pdg_group[abs(part_pdg_group[:, 0]) == 13][:, 1])


		# column 5 is "cluster ID", which is the GEANT4 ID,
		# but unfortunately, our current overlay files have GEANT4 interactions
		# that were generated separately overlaid on each other.
		# column 6 is "group ID", which is usable.
		muon_vox = {}
		for mu_id in true_muon_groups:
			vox = utility_functions.find_matching_rows(numpy.array(vals["input_data"][:, :3]),
			                                           numpy.array(vals["cluster_label"][vals["cluster_label"][:, 6] == mu_id, :3]))[0]
			if len(vox) > 0:
				muon_vox[mu_id] = vox

		vals[key] = muon_vox

#	print("returning from true_muon_voxidxs_by_cluster():", vals[key])
	return vals[key]
