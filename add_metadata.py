import argparse
import glob

import numpy as np
import h5py

from constants import *

def update_data(h5f, dset_name, data):
    if dset_name in h5f:
        del h5f[dset_name]
    h5f[dset_name] = data

def add_metadata(run, verbose=False):
    energies = np.empty((0,))
    nbunches = np.empty((0,), dtype='i4')
    tids = np.empty((0,), dtype='u8')
    flist = sorted(glob.glob(PREFIX + 'raw/r%.4d/'%run + DOOCS_FGLOB))
    for fname in flist:
        with h5py.File(fname, 'r') as f:
            lam = f[WAVELENGTH_DSET][:]
            lam[lam==0] = 1e-6
            energies = np.append(energies, 1239.84/lam)
            nbunches = np.append(nbunches, f[NBUNCHES_DSET][:])
            tids = np.append(tids, f[TRAINID_DSET][:])
    ntrains = len(tids)
    energies = energies[:ntrains]
    nbunches = nbunches[:ntrains]
    if verbose:
        print(ntrains, 'trains in run', run)
        print('Photon energy range:    ', energies.min(), energies.max())
        print('Num Bunches range:      ', nbunches.min(), nbunches.max())

    gaps = np.empty((0,))
    detdists = np.empty((0,))
    gains = np.empty((0,), dtype='i4')
    flist = sorted(glob.glob(PREFIX + 'raw/r%.4d/'%run + DET_META_FGLOB))
    for fname in flist:
        with h5py.File(fname, 'r') as f:
            motorpos = np.array([f[dset_name][:] for dset_name in MOTOR_DSETS])

            detdists = np.append(detdists, 367 - motorpos[4])
            gaps = np.append(gaps, motorpos[0] + motorpos[1] - 33.58)
            gains = np.append(gains, f[GAIN_DSET][:])
    gaps = gaps[:ntrains]
    detdists = detdists[:ntrains]
    gains = gains[:ntrains]
    if verbose:
        print('Detector gap range:     ', gaps.min(), gaps.max())
        print('Detector distance range:', detdists.min(), detdists.max())
        print('Gain mode range:        ', gains.min(), gains.max())

    with h5py.File(PREFIX + 'ayyerkar/data/events/r%.4d_events.h5'%run, 'a') as f:
        update_data(f, 'entry_1/trainId', tids)
        update_data(f, 'entry_1/photon_energy_eV', energies)
        update_data(f, 'entry_1/num_bunches', nbunches)
        update_data(f, 'entry_1/detector_gap_mm', gaps)
        update_data(f, 'entry_1/detector_dist_mm', detdists)
        update_data(f, 'entry_1/detector_gain_mode', gains)

def main():
    parser = argparse.ArgumentParser(description='Calculate event-wise litpixels')
    parser.add_argument('run', help='Run number', type=int)
    parser.add_argument('-v', '--verbose', help='Verbose', action='store_true')
    args = parser.parse_args()

    add_metadata(args.run, verbose=args.verbose)

if __name__ == '__main__':
    main()
