from vibrav.vibronic import Vibronic
from vibrav.base import resource
from vibrav.util.open_files import open_txt
import numpy as np
import pandas as pd
import tarfile
import os
import shutil
import pytest

@pytest.mark.parametrize('freqdx', [[1,7,8], [0], [-1], [15,3,6]])
def test_vibronic_coupling(freqdx):
    with tarfile.open(resource('molcas-ucl6-2minus-vibronic-coupling.tar.xz'), 'r:xz') as tar:
        tar.extractall()
    parent = os.getcwd()
    os.chdir('molcas-ucl6-2minus-vibronic-coupling')
    vib = Vibronic(config_file='va.conf')
    vib.vibronic_coupling(property='electric_dipole', print_stdout=False, temp=298,
                          write_property=False, write_oscil=True, boltz_states=2,
                          write_energy=False, verbose=False, eq_cont=False, select_fdx=freqdx)
    base_oscil = open_txt(resource('molcas-ucl6-2minus-oscillators.txt.xz'), compression='xz',
                          rearrange=False)
    test_oscil = open_txt(os.path.join('vibronic-outputs', 'oscillators-0.txt'), rearrange=False)
    test_oscil = test_oscil[np.logical_and(test_oscil['oscil'].values > 0,
                                           test_oscil['energy'].values > 0)]
    cols = ['oscil', 'energy']
    if freqdx[0] == -1:
        freqdx = range(15)
    print(base_oscil.groupby('freqdx').filter(lambda x: x['freqdx'].unique() in freqdx)[cols].head().to_string())
    #print(test_oscil.groupby('sign').filter(lambda x: x['sign'].unique() in ['minus', 'plus']).head().to_string())
    base = base_oscil.groupby('freqdx').filter(lambda x: x['freqdx'].unique()
                                                         in freqdx)
    base.sort_values(by=['freqdx', 'sign', 'nrow', 'ncol'], inplace=True)
    base = base[cols].values
    test = test_oscil.groupby('sign').filter(lambda x: x['sign'].unique()
                                                       in ['minus', 'plus'])
    test.sort_values(by=['freqdx', 'sign', 'nrow', 'ncol'], inplace=True)
    print(test.head().to_string())
    test = test[cols].values
    print(np.where(np.logical_not(np.isclose(base[:, 0], test[:, 0]))))
    print(base[np.where(np.logical_not(np.isclose(base, test)))[0],0])
    print(np.sort(base[np.where(np.logical_not(np.isclose(base, test)))[0],0]))
    print(test[np.where(np.logical_not(np.isclose(base, test)))[0],0])
    print(np.sort(test[np.where(np.logical_not(np.isclose(base, test)))[0],0]))
    print(len(np.where(np.logical_not(np.isclose(base[:, 0], test[:, 0])))[0]))
    print(base.shape, test.shape)
    assert np.allclose(base[:,0], test[:,0])
    assert np.allclose(base[:,1], test[:,1])
    # test that the individual components average to the isotropic value
    sum_oscil = np.zeros(base.shape[0])
    for idx in range(1, 4):
        df = open_txt(os.path.join('vibronic-outputs', 'oscillators-{}.txt'.format(idx)),
                      rearrange=False)
        df = df[np.logical_and(df['oscil'].values > 0, df['energy'].values > 0)]
        df.sort_values(by=['freqdx', 'sign', 'nrow', 'ncol'], inplace=True)
        sum_oscil += df['oscil'].values
    sum_oscil /= 3.
    assert np.allclose(base[:,0], sum_oscil)
    os.chdir(parent)
    shutil.rmtree('molcas-ucl6-2minus-vibronic-coupling')

