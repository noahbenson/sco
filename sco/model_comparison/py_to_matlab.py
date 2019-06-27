####################################
# py_to_matlab.py
#
# small script that contains utility functions for getting information back and forth between
# python and matlab. For now, contains small functions that take in an array (or a file containing
# values) and increments or decrements the values by one.
#
# by William F. Broderick

import argparse
import numpy as np


def increment_values(values):
    """This silly little function increments the values of a numerical vector by one

    values can be either a vector, in which case they are incremented directly, or a string, in
    which case it's assumed that the string is the path to a txt file you want us to open, increase
    all the values by one, and overwrite. Either way, values (either the new vector or the path to
    the file) is returned
    """
    if pimms.is_str(values):
        with open(values, 'r') as f:
            values_tmp = f.read()
        values_tmp = np.asarray([int(i) for i in values_tmp.strip().split(' ')])+1
        values_tmp.tofile(values, ' ')
        return values
    else:
        return np.asarray(values)+1

def decrement_values(values):
    """This silly little function decrements the values of a numerical vector by one

    values can be either a vector, in which case they are decremented directly, or a string, in
    which case it's assumed that the string is the path to a txt file you want us to open, increase
    all the values by one, and overwrite. Either way, values (either the new vector or the path to
    the file) is returned
    """
    if pimms.is_str(values):
        with open(values, 'r') as f:
            values_tmp = f.read()
        values_tmp = np.asarray([int(i) for i in values_tmp.strip().split(' ')])-1
        values_tmp.tofile(values, ' ')
        return values
    else:
        return np.asarray(values)-1

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-p2m", "--python_to_matlab", help=("string. path to file containing indices"
                                                           " you want to increment by 1"),
                        default=None)
    parser.add_argument("-m2p", "--matlab_to_python", help=("string. path to file containing indices"
                                                           " you want to decrement by 1"),
                        default=None)
    args = parser.parse_args()
    if args.matlab_to_python is not None:
        decrement_values(args.matlab_to_python)
    if args.python_to_matlab is not None:
        increment_values(args.python_to_matlab)
