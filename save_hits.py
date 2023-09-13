import sys
import os.path as op
import argparse
import glob

import numpy as np
import h5py
import dragonfly

from get_thresh import get_thresh
from constants import PREFIX, ADU_PER_KEV, DET_DSET

def cmod(fr):
    out = np.empty_like(fr)
    out[:512] = fr[:512] - np.median(fr[:512], axis=0, keepdims=True)
    out[512:] = fr[512:] - np.median(fr[512:], axis=0, keepdims=True)
    return out

def main():
    parser = argparse.ArgumentParser(description='Save hits to emc file')
    parser.add_argument('run', help='Run number', type=int)
    parser.add_argument('dark_run', help='Associated dark run', type=int)
    parser.add_argument('-E', '--photon_energy', help='Photon energy in eV (default: 2000)', type=float, default=2000)
    parser.add_argument('-t', '--threshold', help='Manual hitscore threshold', type=float)
    args = parser.parse_args()
    
    dark_fname = glob.glob(PREFIX + 'ayyerkar/data/dark/r%.4d*.h5'%args.dark_run)[0]
    with h5py.File(dark_fname, 'r') as f:
        offset = f['offset'][:]
        gain_mode = int(f['gain_mode'][...])

    mask = np.load(PREFIX + 'ayyerkar/data/mask_nan.npy')
    mask[np.isnan(mask)] = 0

    adu_per_photon = ADU_PER_KEV * (args.photon_energy/1000) * (1/gain_mode)

    thresh, hit_inds = get_thresh(args.run, threshold=args.threshold, update=True)
    f_num, f_ind = np.unravel_index(hit_inds, (1000, 500))
    
    wemc = dragonfly.EMCWriter(PREFIX+'ayyerkar/data/emc/r%.4d.emc'%args.run, 1024**2, hdf5=False)
    print('Saving %d hits' % len(hit_inds))
    
    for s in np.unique(f_num):
        raw_fname = PREFIX+'raw/r%.4d/RAW-R%.4d-PNCCD01-S%.5d.h5' % (args.run, args.run, s)
        s_ind = f_ind[f_num==s]
        with h5py.File(raw_fname, 'r') as f:
            dset = f[DET_DSET]
            for i in s_ind:
                frame = cmod(dset[i] - offset) * mask
                phot = (frame/adu_per_photon + 0.3).astype('i4')
                phot[phot<0] = 0
                wemc.write_frame(phot.ravel())
                sys.stderr.write('\r%s: %4d'%(op.basename(raw_fname), i))
    sys.stderr.write('\n')

    wemc.finish_write()

if __name__ == '__main__':
    main()
