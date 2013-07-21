%module fast_agg

%{
#define SWIG_FILE_WITH_INIT
#include "fast_agg.h"
%}

/* Standard numpy setup */
%include "numpy.i"
%init %{
import_array();
%}

%apply (double* IN_ARRAY2, int DIM1, int DIM2) {(double* data, int n, int p)};
%apply (double* ARGOUT_ARRAY1, int DIM1) {(double* means, int m),
                                          (double* stdevs, int s),
                                          (double* ess, int e),
                                          (double* medians, int m)};

%include "fast_agg.h"

%pythoncode %{
def col_mean_std(x):
    """
    Compute mean and standard deviation of each column in matrix using Welford's
    algorithm. Uses no additional heap memory above what's needed for the final
    results.
    """
    if len(x.shape) != 2:
        raise ValueError("Need a matrix")
    return _fast_agg.ColMeanStdevs(x, x.shape[1], x.shape[1])

def effective_sample_sizes(x):
    if len(x.shape) != 2:
        raise ValueError("Need a matrix")
    return _fast_agg.ColEffectiveSampleSizes(x, x.shape[1])
    
def col_medians(x):
    if len(x.shape) != 2:
        raise ValueError("Need a matrix")
    return _fast_agg.ColMedians(x, x.shape[1])
%}
