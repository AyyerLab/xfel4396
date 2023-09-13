import sys
import os.path as op
import argparse
import glob
import time

import numpy as np
import h5py

from constants import PREFIX, DET_FGLOB, DET_DSET, TRAINID_DSET

def get_litpix(run, offset, mask, gain_mode=16, photon_energy=2000):
    adu_thresh = 380 * (16/gain_mode) * (photon_energy/2000)
    litpix = np.empty((0,))
    train_ids = np.empty((0,), dtype='u8')
    flist = sorted(glob.glob(PREFIX + 'raw/r%.4d/'%run + DET_FGLOB))
    for fnum, fname in enumerate(flist):
        with h5py.File(fname, 'r') as f:
            dset = f[DET_DSET]
            medval = np.median(dset[0] - offset)
            offset += medval

            f_litpix = np.zeros((dset.shape[0]))
            for i in range(dset.shape[0]):
                if dset[i,0].sum() == 0:
                    continue
                frame = dset[i] - offset
                f_litpix[i] = (frame*mask > adu_thresh).sum()
                #sys.stderr.write('\r%s: %d/%d    ' % (op.basename(fname), i+1, dset.shape[0]))

            litpix = np.append(litpix, f_litpix)
            train_ids = np.append(train_ids, f[TRAINID_DSET][:])
    #sys.stderr.write('\n')
    return litpix, train_ids

def main():
    parser = argparse.ArgumentParser(description='Calculate event-wise litpixels')
    parser.add_argument('run', help='Run number', type=int)
    parser.add_argument('dark_run', help='Dark run number', type=int)
    parser.add_argument('-E', '--photon_energy', help='Photon energy in eV (default: 2000)', type=float, default=2000)
    args = parser.parse_args()

    dark_fname = glob.glob(PREFIX + 'ayyerkar/data/dark/r%.4d_*.h5' % args.dark_run)[0]
    with h5py.File(dark_fname, 'r') as f:
        offset = f['offset'][:]
        gain_mode = int(f['gain_mode'][...])
    mask = np.load(PREFIX + 'ayyerkar/data/mask_nan.npy')

    stime = time.time()
    litpix, train_ids = get_litpix(args.run, offset, mask,
                                   gain_mode=gain_mode,
                                   photon_energy=args.photon_energy)
    print('Processed run %d in %.3f s' % (args.run, time.time() - stime))

    with h5py.File(PREFIX+'ayyerkar/data/events/r%.4d_events.h5'% args.run, 'w') as f:
        f['entry_1/litpixels'] = litpix
        f['entry_1/trainId'] = train_ids

if __name__ == '__main__':
    main()
