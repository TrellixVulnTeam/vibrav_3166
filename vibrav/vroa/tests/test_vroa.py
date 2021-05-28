from vibrav.vroa import VROA
from vibrav.base import resource
import numpy as np
import pandas as pd
import pytest
import bz2
import lzma
import os
import tarfile
import shutil


def test_vroa():
    with tarfile.open(resource('h2o2-vroa-nwchem.tar.xz'), 'r:xz') as tar:
        tar.extractall()
    homedir = os.getcwd()
    os.chdir('h2o2-vroa-nwchem')
    cls = VROA(config_file='va.conf')
    cls.vroa(raman_units=False, temp=None)
    scat = pd.read_csv('scatter.csv')
    ramn = pd.read_csv('raman.csv')
    # check the ROA data
    for idx, data in cls.scatter.groupby('exc_idx'):
        test = scat.groupby('exc_idx').get_group(idx)
        # check backscattering intensities
        assert np.allclose(test.backscatter.values, data.backscatter.values)
        # check forwardscatter intensities
        assert np.allclose(test.forwardscatter.values, data.forwardscatter.values)
    # check the raman data
    for idx, data in cls.raman.groupby('exc_idx'):
        test = ramn.groupby('exc_idx').get_group(idx)
        assert np.allclose(test.raman_int.values, data.raman_int.values)
    os.chdir(homedir)
    shutil.rmtree('h2o2-vroa-nwchem')
