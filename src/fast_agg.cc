#include <math.h>
#include <vector>
#include <algorithm>
#include "fast_agg.h"

void ColMeanStdevs(double* data, int n, int p,
                   double* means, int m,
                   double* stdevs, int s) {
    /* Iterate over columns */
    for (int j=0; j < p; j++) {
        means[j] = 0.;
        stdevs[j] = 0.; // Holds sum of squares for now
        for (int i=0; i < n; i++) {
            double delta = data[p*i + j] - means[j]; // Assuming row-major order
            means[j] += delta / (i + 1);
            stdevs[j] += delta * (data[p*i + j] - means[j]);
        }
        // Correct denominator and sqrt to get std deviation
        stdevs[j] = sqrt(stdevs[j] / (n - 1));
    }
}

double EffectiveSampleSize(double* x, int n, int offset, int stride) {
    // Compute mean and variance simultanously
    double mean=0., var=0., rho=0.;
    for (int i=0; i < n; i++) {
        double delta = x[offset + i*stride] - mean;
        mean += delta / (i + 1);
        var += delta * (x[offset + i*stride] - mean);
    }
    // Correct denominator and sqrt to get std deviation
    var /= (n - 1);

    // Compute first-order autocorrelation using these values
    for (int i=1; i < n; i++) {
        rho += (x[offset + i*stride] - mean) / var *
            (x[offset + (i-1)*stride] - mean) / (n-1);
    }

    return n * (1 - rho) / (1 + rho);
}

void ColEffectiveSampleSizes(double* data, int n, int p,
                             double* ess, int e) {
    // Iterate over columns and obtain effective sample size for each one
    for (int j=0; j < p; j++) {
        ess[j] = EffectiveSampleSize(data, n, j, p);
    }
}

void ColMedians(double* data, int n, int p,
                double* medians, int m) {
    // Determine last index to sort
    int sort_through = n / 2 + 1;

    // Iterate over columns
    std::vector<double> col (n, 0.);
    for (int j=0; j < p; j++) {
        // Copy column to std::vector for partial sort
        for (int i=0; i < n; i++) {
            col[i] = data[p*i + j]; // Assuming row-major order
        }

        // Partially sort to obtain median
        std::partial_sort(col.begin(), col.begin() + sort_through, col.end());

        // Compute the median
        medians[j] = col[(n - 1) / 2] / 2. + col[n / 2] / 2.;
    }
}
