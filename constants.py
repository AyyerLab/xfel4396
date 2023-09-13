'''Constants to be used during beamtime'''
PREFIX = '/gpfs/exfel/exp/SQS/202302/p004396/scratch/'

DET_FGLOB = 'RAW-R*-PNCCD01-S*.h5'
DET_DSET = 'INSTRUMENT/SQS_NQS_PNCCD1MP/CAL/PNCCD_FMT-0:output/data/image'
TRAINID_DSET = 'INDEX/trainId'

DET_META_FGLOB = 'RAW-R*-PNCCD02-S*.h5'
GAIN_DSET = '/CONTROL/SQS_NQS_PNCCD1MP/MDL/DAQ_GAIN/pNCCDGain/value'
MOTOR_TYPES = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'FEL']
MOTOR_DSETS = ['/CONTROL/SQS_NQS_PNCCD/MOTOR/PNCCD_%s/actualPosition/value'%mtype for mtype in MOTOR_TYPES]

DOOCS_FGLOB = 'RAW-R*-DA01-S*.h5'
WAVELENGTH_DSET = '/CONTROL/SQS_DIAG1_XGMD/XGM/DOOCS/pulseEnergy/wavelengthUsed/value'
NBUNCHES_DSET = '/CONTROL/SQS_DIAG1_XGMD/XGM/DOOCS/pulseEnergy/numberOfSa3BunchesActual/value'

ADU_PER_KEV = 4520 # in Gain mode 1
