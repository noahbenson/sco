####################################################################################################
# sco/test/core.py
# Core tests for the standard cortical observer (SCO) library.
# By Noah C. Benson

import unittest, os, sys, six, logging, pimms
import numpy      as np
import scipy      as sp
import pyrsistent as pyr
import neuropythy as ny

from ..impl.testing import sco_plan as scp_plan_testing

def extract_roi(data, area, hemi=None):
    '''
    extract_roi(data, area) extracts the ROI with the given area id (1, 2, or 3) from the given imap
      data, which must have been produced from an SCO model. The return value is a numpy array of
      the indices of the ROI.

    The optional argument hemi may be supplied to specify that the given hemisphere should also be
    selected.
    '''
    pass
