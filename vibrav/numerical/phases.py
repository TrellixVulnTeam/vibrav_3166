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
from numba import jit, vectorize, float64
import pandas as pd
import numpy as np

@vectorize([float64(float64, float64)])
def get_mag(real, imag):
    return np.sqrt(np.square(real) + np.square(imag))

@jit(nopython=True)
def get_theta(real, imag, state, exc, nexc, ncomps, tol):
    r2max = 0
    imax = 1
    # determine which component is the largest and select that
    # on to make it uniquely real
    for ix in range(ncomps):
        r2 = real[ix*nexc+exc]**2 + imag[ix*nexc+exc]**2
        if r2 > r2max:
            r2max = r2
            imax = ix
    # store the magnitude
    rmax = np.sqrt(r2max)
    # check if the chosen value is storngly imaginary as we would have
    # a division by zero
    # if that is the case the phase angle would be 90 degrees
    if np.abs(real[imax*nexc+exc]) > tol:
        theta = -np.arctan2(imag[imax*nexc+exc], real[imax*nexc+exc])
    else:
        theta = -np.pi/2
    maxdip = (real[imax*nexc+exc] + 1j*imag[imax*nexc+exc])*np.exp(1j*theta)
    if np.real(maxdip) < 0: theta += np.pi
    if np.abs(np.abs(theta) - np.pi) < tol: theta = 0.0
    # write the values to the theta array
    return theta, imax, rmax

@jit(nopython=False)
def insert_values(arr, idx, mag, ncol, nrow, comp, prop):
    arr[idx][0] = np.real(mag)
    arr[idx][1] = np.imag(mag)
    arr[idx][2] = np.sqrt(np.real(mag*np.conjugate(mag)))
    arr[idx][3] = ncol
    arr[idx][4] = nrow
    arr[idx][5] = comp
    arr[idx][6] = prop

@jit(nopython=True)
def perform_phase(dip_real, dip_imag, mag_real, mag_imag,
                  quad_real, quad_imag, nexc, state):
    tol = 1e-6
    # thetas is nexc x 6
    # colums are theta, nrow, ncol, imax, rmax (magnitude of
    # max index), prop (property that we are using for theta)
    thetas = np.zeros((nexc,6))
    # all arrays are nexc*ncomps x 7
    # ncomps is 3 for dipole and angmom properties and 6 for
    # the quadrupole
    # columns are real, imag, magnitude, ncol, nrow, component,
    # prop (identifier for the different properties)
    # the identifier is 0: dipole, 1: angmom, 2: quadrupole
    dip_arr = np.zeros((nexc*3,7))
    mag_arr = np.zeros((nexc*3,7))
    quad_arr = np.zeros((nexc*6,7))
    for exc in range(nexc):
        # if exc == state: continue
        # get the magnitudes of the values at the given
        # excitation for each component
        transdip = get_mag(dip_real[exc::nexc], dip_imag[exc::nexc])
        transmag = get_mag(mag_real[exc::nexc], mag_imag[exc::nexc])
        transquad = get_mag(quad_real[exc::nexc], quad_imag[exc::nexc])
        # detrmine which has a value greater than 1e-4
        #
        # if none do then we fill the column indeces 3, 4, and 5
        # with the exc, state, and ix information, respectively
        # where exc, state, and ix are the ncol, nrow, and component
        # values
        # all other values are zero
        if transdip.max() > 1e-5:
            real = dip_real
            imag = dip_imag
            ncomps = 3
            prop = 0
        elif transmag.max() > 1e-5:
            real = mag_real
            imag = mag_imag
            ncomps = 3
            prop = 1
        elif transquad.max() > 1e-5:
            real = quad_real
            imag = quad_imag
            ncomps = 6
            prop = 2
        else:
            for ix in range(3):
                insert_values(dip_arr, exc+ix*nexc, 0, exc,
                              state, ix, 0)
                insert_values(mag_arr, exc+ix*nexc, 0, exc,
                              state, ix, 1)
            for ix in range(6):
                insert_values(quad_arr, exc+ix*nexc, 0, exc,
                              state, ix, 2)
            continue
        theta, rmax, imax = get_theta(real, imag, state, exc,
                                      nexc, ncomps, tol)
        thetas[exc][0] = theta
        thetas[exc][1] = state
        thetas[exc][2] = exc
        thetas[exc][3] = rmax
        thetas[exc][4] = imax
        thetas[exc][5] = prop
        # remove the phase from each of the properties
        # perform two for loops for the dipole and angmom values
        # in the first loop simultaneously
        # second loop is for the quadrupole values
        for ix in range(3):
            maxdip = (dip_real[ix*nexc+exc] + 1j*dip_imag[ix*nexc+exc]) \
                        *np.exp(1j*theta)
            insert_values(dip_arr, exc+ix*nexc, maxdip, exc, state, ix, 0)
            maxdip = (mag_real[ix*nexc+exc] + 1j*mag_imag[ix*nexc+exc]) \
                        *np.exp(1j*theta)
            insert_values(mag_arr, exc+ix*nexc, maxdip, exc, state, ix, 1)
        for ix in range(6):
            maxdip = (quad_real[ix*nexc+exc] + 1j*quad_imag[ix*nexc+exc]) \
                        *np.exp(1j*theta)
            insert_values(quad_arr, exc+ix*nexc, maxdip, exc, state, ix, 2)
    return dip_arr, mag_arr, quad_arr, thetas

def transform_array_to_df(arr, cols, int_cols, prop_map):
    df = pd.DataFrame(arr, columns=cols)
    df[int_cols] = df[int_cols].astype(int)
    # map the property elements to their string values
    df['prop'] = df['prop'].map(prop_map)
    df['component'] += 1
    return df

def correct_phase_angle(data_df):
    '''
    Here we analyze the phase angle of the input data and perform a
    "rotation" by the phase angle. We determine the phase angle by
    getting the highest absolute magnitude in each "transition"
    (`nrow` value) and determine the angle such that we minimize
    the imaginary value for that component in the "transition".

    Note:
        This has been hard-coded for Molcas/OpenMolcas with the
        files printed from RASSI with the `PRPRint` keyword.
    '''
    dfs = []
    thetas = []
    # set the columns that will be integers after the
    # function where we rotate the data by the phase
    int_cols = ['ncol', 'nrow', 'component', 'prop']
    # mapping dictionaries
    kwarg_map = {'dip': 'dipole', 'mag': 'angmom',
                 'quad': 'quadrupole'}
    prop_map = {0: 'dipole', 1: 'angmom', 2: 'quadrupole'}
    for state, gs in data_df.groupby('nrow'):
        group = gs.groupby('prop').get_group
        # generate the arrays that are needed by the
        # function and store as dictionary with the
        # proper kwarg names
        kwargs = {}
        for key in kwarg_map.keys():
            for t in ['real', 'imag']:
                kwargs[key+'_'+t] = group(kwarg_map[key])[t].values
        kwargs['nexc'] = group('dipole')['ncol'].unique().shape[0]
        kwargs['state'] = state
        # rotate data by the phase angle
        arrays = perform_phase(**kwargs)
        # get the arrays that store the property data and
        # and organize it respectively
        cols = ['real', 'imag', 'magnitude', 'ncol', 'nrow', 'component', 'prop']
        df = transform_array_to_df(np.concatenate(arrays[:3]), cols,
                                   int_cols, prop_map)
        dfs.append(df)
        cols = ['theta', 'nrow', 'ncol', 'component', 'magnitude', 'prop']
        df = transform_array_to_df(arrays[3], cols, int_cols, prop_map)
        thetas.append(df)
    rot_df = pd.concat(dfs, ignore_index=True)
    theta = pd.concat(thetas, ignore_index=True)
    return rot_df, theta

