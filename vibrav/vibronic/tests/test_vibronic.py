# This file is part of vibrav.
#
# vibrav is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# vibrav is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with vibrav.  If not, see <https://www.gnu.org/licenses/>.
from vibrav.vibronic import Vibronic
from vibrav.base import resource
from vibrav.util.io import open_txt
import numpy as np
import pandas as pd
import tarfile
import os
import shutil
import pytest

@pytest.mark.parametrize('freqdx', [[1,7,8], [0], [-1], [15,3,6]])
def test_vibronic_coupling(freqdx):
    with tarfile.open(resource('molcas-ucl6-2minus-vibronic-coupling.tar.xz'), 'r:xz') as tar:
        def is_within_directory(directory, target):
            
            abs_directory = os.path.abspath(directory)
            abs_target = os.path.abspath(target)
        
            prefix = os.path.commonprefix([abs_directory, abs_target])
            
            return prefix == abs_directory
        
        def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
        
            for member in tar.getmembers():
                member_path = os.path.join(path, member.name)
                if not is_within_directory(path, member_path):
                    raise Exception("Attempted Path Traversal in Tar File")
        
            tar.extractall(path, members, numeric_owner=numeric_owner) 
            
        
        safe_extract(tar)
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
    base = base_oscil.groupby('freqdx').filter(lambda x: x['freqdx'].unique()
                                                         in freqdx)
    base.sort_values(by=['freqdx', 'sign', 'nrow', 'ncol'], inplace=True)
    base = base[cols].values
    test = test_oscil.groupby('sign').filter(lambda x: x['sign'].unique()
                                                       in ['minus', 'plus'])
    test.sort_values(by=['freqdx', 'sign', 'nrow', 'ncol'], inplace=True)
    test = test[cols].values
    assert np.allclose(base[:,0], test[:,0], rtol=7e-5)
    assert np.allclose(base[:,1], test[:,1], rtol=1e-5, atol=1e-7)
    # test that the individual components average to the isotropic value
    #sum_oscil = np.zeros(base.shape[0])
    #for idx in range(1, 4):
    #    df = open_txt(os.path.join('vibronic-outputs', 'oscillators-{}.txt'.format(idx)),
    #                  rearrange=False)
    #    df = df[np.logical_and(df['oscil'].values > 0, df['energy'].values > 0)]
    #    df.sort_values(by=['freqdx', 'sign', 'nrow', 'ncol'], inplace=True)
    #    sum_oscil += df['oscil'].values
    #sum_oscil /= 3.
    #assert np.allclose(base[:,0], sum_oscil)
    os.chdir(parent)
    shutil.rmtree('molcas-ucl6-2minus-vibronic-coupling')

