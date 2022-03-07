"""
  truth_functions.py : Helper functions that don't fit into another category

  Original author:  J. Wolcott <jwolcott@fnal.gov>
             Date:  Mar. 2022
"""

import numpy


def find_matching_rows(a1, a2, mask_only=False):
	""" Find the indices of rows of 2D array a1 that match to rows of 2D array a2
	"""
	assert(a1.dtype == a2.dtype)

	# the 1D case is straightforward
	if len(a1.shape) == 1:
		return numpy.argwhere(numpy.isin(a1, a2))

	assert(a1.shape[1] == a2.shape[1])

	# adapted from https://stackoverflow.com/questions/16210738/implementation-of-numpy-in1d-for-2d-arrays
	dtype = ",".join([str(a1.dtype),] * a1.shape[1])
	# this gives us a mask for the first array
	mask = numpy.in1d(a1.view(dtype=dtype).reshape(a1.shape[0]), a2.view(dtype=dtype))
	if mask_only:
		return mask
	else:
		return numpy.nonzero(mask)

	# solution below is quite slow for large arrays due to all the broadcasting
	# (adapted from adapted from https://stackoverflow.com/questions/64930665/find-indices-of-rows-of-numpy-2d-array-in-another-2d-array)
	# # 2D array where each row corresponds to a row from a2,
	# # and each column corresponds to a row from a1.
	# # matching rows should have had a row of shape[1]*True
	# # from the equality test; the all() combines them for easy testing
	# matches_by_a2row = (a1 == a2[:, None]).all(axis=2)
	#
	# # the argwhere returns an array of pairs [(a2 row index, a1 row index), (a2 row index, a1 row index), ...]
	# return numpy.argwhere(matches_by_a2row)[:, 1]
