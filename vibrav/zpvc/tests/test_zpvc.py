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
from vibrav.zpvc import ZPVC
from vibrav.base import resource
import numpy as np
import pandas as pd
import pytest
import tarfile
import os
import glob

@pytest.fixture
def zpvc_results():
    df = pd.read_csv(resource('nitromalonamide-zpvc-results.csv.xz'), compression='xz',
                     index_col=False)
    zpvc_results = df.groupby('temp')
    yield zpvc_results

@pytest.fixture
def zpvc_geometry():
    df = pd.read_csv(resource('nitromalonamide-zpvc-geometry.csv.xz'), compression='xz',
                     index_col=False)
    zpvc_geometry = df.groupby('temp')
    yield zpvc_geometry

@pytest.fixture
def grad(nat):
    df = pd.read_csv(resource('nitromalonamide-zpvc-grad.dat.xz'), compression='xz',
                     index_col=False, header=None)
    tmp = df.values.reshape(nat*((nat*3-6)*2+1), 3).T
    grad = pd.DataFrame.from_dict({'fx': tmp[0], 'fy': tmp[1], 'fz': tmp[2]})
    grad['file'] = np.repeat(range((nat*3-6)*2+1), nat)
    yield grad

@pytest.fixture
def prop():
    df = pd.read_csv(resource('nitromalonamide-zpvc-prop.dat.xz'), compression='xz',
                     index_col=False, header=None)
    df['file'] = df.index
    prop = df.copy()
    yield prop

@pytest.mark.parametrize("temp, nat", [([0], 15), ([100], 15), ([200], 15), ([300], 15),
                                       ([400], 15), ([600], 15)])
def test_zpvc(zpvc_results, zpvc_geometry, grad, prop, temp):
    with tarfile.open(resource('nitromalonamide-zpvc-dat-files.tar.xz'), 'r:xz') as tar:
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
    zpvc = ZPVC(config_file=resource('nitromalonamide-zpvc-config.conf'))
    zpvc.zpvc(gradient=grad, property=prop, temperature=temp, write_out_files=False)
    data_files = glob.glob('*.dat')
    for file in data_files: os.remove(file)
    test_cols = ['tot_anharm', 'tot_curva', 'zpvc', 'property', 'zpva']
    exp_cols = ['anharm' ,'curv' ,'zpvc' ,'prop' ,'zpva']
    print(zpvc.vib_average.to_string())
    print(zpvc_results.get_group(temp[0])[exp_cols].values)
    print(zpvc.zpvc_results[test_cols].values)
    assert np.allclose(zpvc_results.get_group(temp[0])[exp_cols].values,
                       zpvc.zpvc_results[test_cols].values, atol=1e-3)

