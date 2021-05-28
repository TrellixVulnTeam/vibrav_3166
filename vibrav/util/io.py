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
import pandas as pd
import numpy as np
import os
import warnings
import re
import lzma

def open_txt(fp, rearrange=True, get_complex=False, fill=False, is_complex=True, tol=None,
             **kwargs):
    '''
    Method to open a .txt file that has a separator of ' ' with the first columns ordered as
    ['nrow', 'ncol', 'real', 'imag']. We take care of adding the two values to generate a
    complex number. The program will take your data and automatically generate a matrix
    of complex values that has the size of the maximum of the unique values in the 'nrow'
    column and the maximum in the 'ncol' column. tha being said if there are missing values
    the program will throw 'ValueError' as it will not be able to determine the size of the
    new matrix. This works for both square and non-square matrices. We assume that the indexing
    is non-pythonic hence the subtraction of 'nrow' and 'ncol' columns.

    Args:
        fp (str): Filepath of the file you want to open.
        sep (str, optional): Delimiter value.
        skipinitialspace (bool, optional): Pandas skipinitialspace argument in the `pandas.read_csv`
                                           method.
        rearrange (bool, optional): If you want to rearrange the data into a square complex matrix.
                                    Defaults to `True`.
        fill (:obj:`bool`, optional): Fill the missing indeces with zeros.
        is_complex (:obj:`bool`, optional): The input data is complex.
        **kwargs (optional): Arguments that will be passed into :code:`pandas.read_csv`

    Returns:
        matrix (pandas.DataFrame): Re-sized complex square matrix with the appropriate size.

    Raises:
        TypeError: When there are null values found. A common place this has been an issue is when
                   the first two columns in the '.txt' files read have merged due to too many states.
                   This was found to happen when there were over 1000 spin-orbit states.
    '''
    keys = kwargs.keys()
    # make sure certain defaults keys are kwargs
    if 'skipinitialspace' not in keys:
        kwargs['skipinitialspace'] = True
    if 'sep' not in keys:
        kwargs['sep'] = ' '
    if 'index_col' not in keys:
        kwargs['index_col'] = False
    df = pd.read_csv(fp, **kwargs)
    if pd.isnull(df).values.any():
        print(np.where(pd.isnull(df)))
        raise TypeError("Null values where found while reading {fp} ".format(fp=fp) \
                       +"a common issue is if the ncol and nrow values have merged")
    # this is an assumption that only works in the context of this program as the
    # txt files that the program is looking for have four columns
    # TODO: might be nice to give a conditional to check if the data is real or complex
    #       and only read the first three columns but that might be more work for nothing
    df.columns = list(map(lambda x: x.lower().replace('#', ''), df.columns))
    df['nrow'] -= 1
    df['ncol'] -= 1
    # rearrange the data to a matrix ordered by the rows and columns
    if rearrange:
        nrow = df['nrow'].unique().shape[0]
        ncol = df['ncol'].unique().shape[0]
        if df.shape[0] == nrow*ncol:
            matrix = pd.DataFrame(df.groupby('nrow').apply(lambda x:
                                  x['real']+1j*x['imag']).values.reshape(nrow, ncol))
        elif fill:
            matrix = np.zeros((nrow, ncol), dtype=np.complex128)
            for row, col, real, imag in zip(df.nrow, df.ncol, df.real, df.imag):
                matrix[row, col] = real + 1j*imag
            matrix = pd.DataFrame(matrix)
        else:
            text = "The given matrix is not square with {} elements and " \
                   +"(nrow, ncol) ({}, {}). If the input matrix is sparse " \
                   +"then use the 'fill=True' as this will fill the matrix " \
                   +"with zeros."
            raise ValueError(text.format(df.shape[0], nrow, ncol))
        if not is_complex:
            real = np.real(matrix.values)
            imag = np.imag(matrix.values)
            if not np.allclose(imag, 0):
                raise ValueError("The input data was detected to be complex " \
                                 +"but the kwarg 'is_complex' was set to " \
                                 +"'False'")
            matrix = pd.DataFrame(np.real(matrix.values))
    else:
        matrix = df.copy()
        if tol is not None:
            index = matrix['real'].abs() < tol
            matrix.loc[index, 'real'] = 0.0
            index = matrix['imag'].abs() < tol
            matrix.loc[index, 'imag'] = 0.0
        if get_complex and not is_complex:
            raise ValueError("Keywords 'get_complex' and 'is_complex' clash " \
                             +"as they were set to 'True' and 'False', " \
                             +"respectively. Cannot calculate the complex " \
                             +"values if they are not complex.")
        if get_complex:
            matrix['complex'] = matrix['real'] + 1j*df['imag']
        if not is_complex:
            if not np.allclose(matrix['imag'].values, 0):
                raise ValueError("The input data was detected to be complex " \
                                 +"but the kwarg 'is_complex' was set to " \
                                 +"'False'")
            matrix.drop('imag', axis=1, inplace=True)
    return matrix

def get_all_data(cls, path, property, f_start='', f_end=''):
    '''
    Function to get all of the data from the files in a specific directory.
    It will look for all of the files that match the given `f_start`
    and `f_end` input parameters and try to extract the information
    for the given `property` with the `cls` parser class.

    Note:
        We recommend that the convention used in creating the different
        files is such that there is an idex to each file to keep track of
        the data that corresponds to the specific file. The program, when
        attempting to find the index of the file, will find all of the
        integers in the filename and assume that the last entry is the
        file index. It will give a warning if it finds more than one
        integer group in the filename to tell the user that it will
        assume that the last group of integers found is the file index.

    Parameters:
        cls (class object): Class object of the output parser of choice.
        path (:obj:`str`): Path to the directory containing all of the
                           output files.
        property (:obj:`str`): Property of interest to parse.
        f_start (:obj:`str`): Starting string to match the output files.
                              Defaults to :code:`''`.
        f_end (:obj:`str`): Ending string to match the output files.
                            Defaults to :code:`''`.

    Returns:
        data (:class:`pandas.DataFrame`): Data frame with all of the parsed data.

    Raises:
        ValueError: If the program cannot find any data or files that
                    match the input parameters.
    '''
    dfs = []
    for (_, _, files) in os.walk(path):
        for file in files:
            filename = os.path.join(path, file)
            if os.path.isfile(filename) and file.startswith(f_start) and file.endswith(f_end):
                ed = cls(filename)
                try:
                    df = getattr(ed, property)
                except AttributeError:
                    print("The property {} cannot be found in the output {}.".format(property, filename))
                    continue
                fdx = list(map(int, re.findall('\d+', file.replace(f_start, '').replace(f_end, ''))))
                if len(fdx) > 1:
                    warnings.warn("More than one index was found in the filename. Will assume that the " \
                                  +"file index is the last number found.", Warning)
                df['file'] = fdx[-1]
            else:
                continue
            dfs.append(df)
    if len(dfs) == 0:
        raise ValueError("No data was found in the directory {}".format(path))
    data = pd.concat(dfs, ignore_index=True)
    return data

def uncompress_file(fp, compression='xz'):
    if compression == 'xz':
        decomp = fp.split(os.sep)[-1][:-3]
        with open(decomp, 'wb') as new_file, lzma.LZMAFile(fp, 'rb') as file:
            for data in iter(lambda : file.read(100 *1024), b''):
                new_file.write(data)
    return decomp
