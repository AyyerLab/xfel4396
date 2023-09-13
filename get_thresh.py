import argparse

import numpy as np
import h5py
from scipy import optimize

from constants import PREFIX

def gaussian(x, a, x0, sigma):
    return a * np.exp(-(x-x0)**2 / 2 / sigma**2)

def update_hit_inds(run, hit_inds):
    print('Updating events file with hit_indices')
    with h5py.File(PREFIX+'ayyerkar/data/events/r%.4d_events.h5'%run, 'a') as f:
        if 'entry_1/hit_indices' in f:
            del f['entry_1/hit_indices']
        f['entry_1/hit_indices'] = hit_inds

def get_thresh(run, lthresh=100, threshold=None, update=False, verbose=False):
    '''Get hit score threshold by fitting Gaussian to background peak and taking a 3-sigma value'''
    # Get lit pixels
    with h5py.File(PREFIX+'ayyerkar/data/events/r%.4d_events.h5'%run, 'r') as f:
        litpix = f['entry_1/litpixels'][:]

    if threshold is not None:
        hit_inds = np.where(litpix > threshold)[0]
        if update:
            update_hit_inds(run, hit_inds)
        return threshold, np.where(litpix>threshold)[0]

    sel_litpix = litpix[litpix > lthresh]

    # Get hit indices
    medval = np.median(sel_litpix)
    if verbose:
        print('Median hit score value =', medval)
    hy, hx = np.histogram(sel_litpix, np.linspace(lthresh, 2*medval, 100))
    hcen = 0.5*(hx[1:] + hx[:-1])
    xmax = hy[1:].argmax() + 1 # Ignoring first bin
    if verbose:
        print('Initial fit params =', (hy.max(), hcen[xmax], hcen[xmax]/10.))
    popt, pcov = optimize.curve_fit(gaussian, hcen[1:xmax], hy[1:xmax], p0=(hy.max(), hcen[xmax], hcen[xmax]/10.))
    threshold = popt[1] + 3*np.abs(popt[2])
    if verbose:
        print('Fitted background Gaussian: %.3f +- %.3f' % (popt[1], popt[2]))

    hit_inds = np.where(litpix > threshold)[0]
    if update:
        update_hit_inds(run, hit_inds)

    return threshold, hit_inds

def main():
    parser = argparse.ArgumentParser(description='Get hit score threshold')
    parser.add_argument('run', help='Run number', type=int)
    parser.add_argument('-v', '--verbose', help='Verbose', action='store_true')
    parser.add_argument('-u', '--update', help='Update hit indices in events file', action='store_true')
    parser.add_argument('-t', '--threshold', help='Manual hitscore threshold', type=float)
    args = parser.parse_args()

    threshold, hit_inds = get_thresh(args.run, threshold=args.threshold, verbose=args.verbose, update=args.update)
    num_hits = hit_inds.size
    print('%d hits using a threshold of %.3f' % (num_hits, threshold))

if __name__ == '__main__':
    main()
