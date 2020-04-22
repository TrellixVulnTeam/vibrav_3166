import numpy as np
import pandas as pd
import warnings
from exa.util.constants import Boltzmann_constant as boltz_constant
from exa.util.units import Energy

def boltz_dist(energies, temp, tol=1e-6, states=None, ignore_max=False):
    '''
    Generate the vibrational boltzmann distributions for all vibrational
    frequencies up to when the distribution is less than the given tolerance
    value.

    Note:
        This does not take into account any energetic degeneracies of the input
        energies/states and the primary purpose is to calculate the Boltzmann distribution
        of vibrational states.

    Args:
        energies (:obj:`numpy.array` or :obj:`list`): Array of energies in wavenumbers.
        temp (:obj:`float`): Temperature to calculate the distribution.
        tol (:obj:`float`, optional): Tolerance threshold to cutoff the calculation
                                      of the distribution factors. Defaults to `1e-5`.
        states (:obj:`int`, optional): Number of states to calculate. Defaults to
                                       `None` (all states with a distribution
                                       less than the `boltz_tol` value for the lowest 
                                       frequency).
        ignore_max (:obj:`bool`, optional): Ignore the default matimum number of max states
                                            to be calculated. This is very dangerous use at
                                            your own risk. There is a possibility of hitting
                                            the recursion limit in python. Defaults to `False`.

    Returns:
        boltz (pd.DataFrame): Data frame of the boltzmann factors for each energy.

    Raises:
        ValueError: If the `states` parameter given is more than 1000 as we consider this
                    to be unphysical.
        ValueError: If the `states` parameter is zero.
        ValueError: If the `temp` paramter is less than or equal to zero.
        ValueError: If the program calculates more than 1000 states at a given tolerance.

    Examples:
        Calculating the boltzmann distribution of a two state system at room temperature
        (298 K) with the energies 10, 20 and 80 wavenumbers (cm^{-1}).

        >>> temp = 298
        >>> freq = [10, 20, 80]
        >>> boltz_dist(energies=freq, temp=temp, states=2)
                  0         1  freqdx  partition  energy
        0  0.512068  0.487932       0   1.952866      10
        1  0.524122  0.475878       1   1.907953      20
        2  0.595379  0.404621       2   1.679601      80

        It is not necessary that the input energies be ordered as shown by the next example.
        The only thing that will change is that the output :obj:`pandas.DataFrame` will have a
        `freqdx` column that will reflect the ordeting of the input energies.

        >>> temp = 298
        >>> freq = [10, 80, 20]
        >>> boltz_dist(energies=freq, temp=temp, states=2)
                  0         1  freqdx  partition  energy
        0  0.512068  0.487932       0   1.952866      10
        1  0.524122  0.475878       2   1.907953      20
        2  0.595379  0.404621       1   1.679601      80

        These examples have truncated the number of states to actually calculate by a user
        defined values. If the user is interested in getting all of the available states above
        a given threshold, 1e-6 by default, this can be done by passing the `states` parameter as
        `None`.

        >>> temp = 40
        >>> freq = [100, 200, 300]
        >>> boltz_dist(energies=freq, temp=temp, states=None)
                  0         1             2             3  freqdx  partition  energy
        0  0.972593  0.026656  7.305767e-04  2.002318e-05       0   1.028179     100
        1  0.999249  0.000751  5.638231e-07  4.235235e-10       1   1.000752     200
        2  0.999979  0.000021  4.238331e-10  8.725631e-15       2   1.000021     300

        Notice, that in the example above, the threshold is only applied to the lowest energy
        and all higher energies calculate the same number of states even when the threshold is
        already met. This is purely done for the sake of simplifying data storage in a later
        step with the `numpy.stack` function. It is not believed that this will affect memory
        usage or performance greatly.

        The program is designed to detect if there are more than 1000 states calculated and
        automatically throw an error as this is an unphysical number of states to calculate.
    '''
    # to simplify code
    def _boltzmann(energ, nu, temp):
        ''' Boltzmann distribution function. '''
        return np.exp(-energ*nu/(boltz_constant*Energy['J', 'cm^-1']*temp))
    # resursive function
    def _partition_func(energ, nu, temp):
        ''' Partition function. '''
        Q = 0
        if nu == 0:
            # terminating condition
            return _boltzmann(energ, nu, temp)
        else:
            # recursive portion
            Q += _boltzmann(energ, nu, temp) + _partition_func(energ, nu-1, temp)
        return Q
    # for single energy values
    # not the default use of this script
    if isinstance(energies, float): energies = [energies]
    def_states = False
    # toss warning if states is not none
    if states is not None:
        # make sure the number of states given is not unphysical
        if states >= 1000 and not ignore_max:
            raise ValueError("An unphysical number of states to calculate was given currently, " \
                             +"{}, expected less than 1000.".format(states))
        warnings.warn("Calculating only the first {} ".format(states) \
                      +"states for the Boltzmann distribution.", Warning)
        max_nu = states
    elif states == 0:
        raise ValueError("The states parameter given was found to be zero. " \
                         "This cannot be understood and a non-sensical value.")
    else:
        # set to a high value that should never be rached
        # can serve as a final failsafe for the while loop
        states = 1e3
        def_states = True
        # ensure that at least two states are calculated
        max_nu = 2
    # make sure the temp is 'real'
    if temp <= 0:
        raise ValueError("A negative or zero temperature was detected. The input temperature " \
                         +"value must be in units of Kelvin and be a non-zero positive value.")
    # reorder the energies in increasing order
    sorted_energies = pd.Series(energies).sort_values()
    boltz_factors = []
    partition = []
    for idx, (fdx, freq) in enumerate(sorted_energies.items()):
        boltz_factors.append([])
        nu = 0
        # using a while loop as it is easier to define multiple
        # termination conditions
        if def_states:
            while (_boltzmann(freq, nu, temp) > tol or nu < max_nu) and nu < 1e3:
                boltz_factors[-1].append(_boltzmann(freq, nu, temp))
                nu += 1
        else:
            while nu < max_nu:
                boltz_factors[-1].append(_boltzmann(freq, nu, temp))
                nu += 1
        if nu >= 1e3 and not ignore_max:
            raise ValueError("There is something wrong with this frequency ({}) and ".format(freq) \
                             +"tolerance ({}) combination ".format(tol) \
                             +"as we have calculated more than 1000 states.")
        # set the maximum number of states to be calculated based on
        # the smallest frequency value to have all same size lists
        if idx == 0 and def_states: max_nu = nu
        # calculate the partition function
        # we subtract 1 from nu because of the termination condition in the while loop
        q = _partition_func(freq, nu-1, temp)
        boltz_factors[-1] /= q
        partition.append(q)
    # put all of the boltzmann factors together
    data = np.stack(boltz_factors, axis=0)
    # make a dataframe for easier data handling
    boltz = pd.DataFrame(data)
    # append some values to keep track of things
    boltz['freqdx'] = sorted_energies.index
    boltz['partition'] = partition
    boltz['energy'] = sorted_energies.values
    return boltz
