####################################################################################################
# sco/impl/kay13/__init__.py
# Implementation details of the SCO specific to the Benson et al. 2017 publications.
# by Noah C. Benson

'''
The sco.impl.kay13 namespace contains default values for all of the optional parameters used in the
implementation of the SCO model that most resembles the version used by Kay et al. (2013). A 
function, provide_default_options is also defined; this may be included in any SCO plan to ensure 
that the optional parameter values are available to the downstream calculations if not provided by
the user or other calculations. In general this namespace strongly resembles the sco.impl.benson17
package.
'''

import pyrsistent              as _pyr
import pimms                   as _pimms
import numpy                   as _np
from   sco.util   import units as _units

import sco.anatomy
import sco.stimulus
import sco.pRF
import sco.contrast
import sco.analysis
import sco.util

from sco.impl.benson17 import (divisively_normalize_Heeger1992,
                               calc_divisive_normalization,
                               pRF_sigma_slopes_by_label_Kay2013,
                               contrast_constants_by_label_Kay2013,
                               compressive_constants_by_label_Kay2013,
                               saturation_constants_by_label_Kay2013,
                               divisive_exponents_by_label_Kay2013)
def cpd_sensitivity(e, s, l):
    '''
    sco.impl.kay13.cpd_sensitivity(ecc, prfsz, lbl) always yields the map {3.0: 1.0}.
    '''
    return {3.0: 1.0}

# Default Options ##################################################################################
# The default options are provided here for the SCO
@_pimms.calc('kay17_default_options_used')
def provide_default_options(
        pRF_sigma_slopes_by_label      = pRF_sigma_slopes_by_label_Kay2013,
        contrast_constants_by_label    = contrast_constants_by_label_Kay2013,
        compressive_constants_by_label = compressive_constants_by_label_Kay2013,
        saturation_constants_by_label  = saturation_constants_by_label_Kay2013,
        divisive_exponents_by_label    = divisive_exponents_by_label_Kay2013,
        max_eccentricity               = 7.5,
        modality                       = 'surface',
        cpd_sensitivity_function       = cpd_sensitivity,
        gabor_orientations             = 8):
    '''
    provide_default_options is a calculator that optionally accepts values for all parameters for
    which default values are provided in the sco.impl.benson17 package and yields into the calc plan
    these parameter values or the default ones for any not provided.
 
    These options are:
      * pRF_sigma_slope_by_label (sco.impl.benson17.pRF_sigma_slope_by_label_Kay2013)
      * compressive_constant_by_label (sco.impl.benson17.compressive_constant_by_label_Kay2013)
      * contrast_constant_by_label (sco.impl.benson17.contrast_constant_by_label_Kay2013)
      * modality ('surface')
      * max_eccentricity (12)
      * cpd_sensitivity_function (sco.impl.benson17.cpd_sensitivity)
      * saturation_constant (sco.impl.benson17.saturation_constant_Kay2013)
      * divisive_exponent (sco.impl.benson17.divisive_exponent_Kay2013)
      * gabor_orientations (8)
    '''
    # the defaults are filled-in by virtue of being in the above argument list
    return True

# The volume (default) calculation chain
sco_plan_data = _pyr.pmap({k:v
                           for pd    in [sco.stimulus.stimulus_plan_data,
                                         sco.contrast.contrast_plan_data,
                                         sco.pRF.pRF_plan_data,
                                         sco.anatomy.anatomy_plan_data,
                                         sco.analysis.analysis_plan_data,
                                         sco.util.export_plan_data,
                                         {'default_options': provide_default_options,
                                          'divisive_normalization': calc_divisive_normalization}]
                           for (k,v) in pd.iteritems()})

sco_plan      = _pimms.plan(sco_plan_data)
