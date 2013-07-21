#ifndef FAST_AGG_H
#define FAST_AGG_H

void ColMeanStdevs(double* data, int n, int p,
                   double* means, int m,
                   double* stdevs, int s);
                
void ColEffectiveSampleSizes(double* data, int n, int p,
                             double* ess, int e);

void ColMedians(double* data, int n, int p,
                double* medians, int m);

#endif
