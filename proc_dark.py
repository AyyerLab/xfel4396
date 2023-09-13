import sys
import os.path as op
import argparse
import glob

import numpy as np
import h5py

from constants import PREFIX, DET_FGLOB, DET_DSET, DET_META_FGLOB, GAIN_DSET

def get_dark(run):
    flist = sorted(glob.glob(PREFIX+'raw/r%.4d/'%run + DET_FGLOB))
    dark = np.zeros((1024,1024))
    darksq = np.zeros((1024,1024))
    nframes = 0
    for fname in flist:
        bfname = op.basename(fname)
        with h5py.File(fname, 'r') as f:
            dset = f[DET_DSET]
            for i in range(dset.shape[0]):
                if dset[i,0,0] == 0.:
                    continue
                fr = dset[i] - 7000.
                dark += fr
                darksq += fr*fr
                nframes += 1
                sys.stderr.write('\r%s: %d/%d' % (bfname, i+1, dset.shape[0]))
            sys.stderr.write('\n')

    dark /= nframes
    darksq /= nframes
    std = np.sqrt(darksq - dark**2)
    return dark + 7000., std

def get_gain_mode(run):
    fname = sorted(glob.glob(PREFIX + 'raw/r%.4d/'%run + DET_META_FGLOB))[0]
    with h5py.File(fname, 'r') as f:
        gain = f[GAIN_DSET][0]
    print('Gain mode: 1/%d' % gain)
    return gain

def main():
    parser = argparse.ArgumentParser(description='Process dark run')
    parser.add_argument('run', help='Run number', type=int)
    args = parser.parse_args()
    
    offset, std = get_dark(args.run)
    gain_mode = get_gain_mode(args.run)
    
    with h5py.File(PREFIX+'ayyerkar/data/dark/r%.4d_g%.2d.h5'%(args.run, gain_mode), 'w') as f:
        f['offset'] = offset
        f['std'] = std
        f['gain_mode'] = gain_mode

if __name__ == '__main__':
    main()
