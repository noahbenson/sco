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
import six
from   sco.util   import units as _units

import sco.anatomy
import sco.stimulus
import sco.pRF
import sco.contrast
import sco.analysis
import sco.util

from sco.impl.benson17 import (pRF_sigma_slopes_by_label_Kay2013,
                               contrast_constants_by_label_Kay2013,
                               compressive_constants_by_label_Kay2013,
                               saturation_constants_by_label_Kay2013,
                               divisive_exponents_by_label_Kay2013,
                               gains_by_label_Kay2013)

def spatial_frequency_sensitivity(prf, cpds):
    '''
    sco.impl.kay13.spatial_frequency_sensitivity(prf, cpds) always yields a narrow band of
      sensitivity to spatial frequencies near 3 cycles/degree.
    '''
    p0 = 3.0
    lf0 = _np.log2(1/p0)
    # weights are log-gaussian distributed around the preferred period
    cpds = _np.log2(_pimms.mag(cpds, 'cycles/degree'))
    ws = _np.exp(-0.5 * ((cpds - lf0)/0.25)**2)
    return ws / _np.sum(ws)

# Default Options ##################################################################################
# The default options are provided here for the SCO
@_pimms.calc('kay17_default_options_used')
def provide_default_options(
        pRF_sigma_slopes_by_label              = pRF_sigma_slopes_by_label_Kay2013,
        contrast_constants_by_label            = contrast_constants_by_label_Kay2013,
        compressive_constants_by_label         = compressive_constants_by_label_Kay2013,
        saturation_constants_by_label          = saturation_constants_by_label_Kay2013,
        divisive_exponents_by_label            = divisive_exponents_by_label_Kay2013,
        gains_by_label                         = gains_by_label_Kay2013,
        max_eccentricity                       = 7.5,
        modality                               = 'surface',
        gabor_spatial_frequencies              = (_pimms.quant(3.0, 'cycles/deg'),),
        spatial_frequency_sensitivity_function = spatial_frequency_sensitivity):
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
      * spatial_frequency_sensitivity_function (sco.impl.kay13.spatial_frequency_sensitivity)
      * saturation_constant (sco.impl.benson17.saturation_constant_Kay2013)
      * divisive_exponent (sco.impl.benson17.divisive_exponent_Kay2013)
      * gabor_orientations (8)
    '''
    # the defaults are filled-in by virtue of being in the above argument list
    return True

# The volume (default) calculation chain
sco_plan_data = _pyr.pmap(
    {k:v
     for pd    in [sco.stimulus.stimulus_plan_data,
                   sco.contrast.contrast_plan_data.discard('gabor_spatial_frequencies'),
                   sco.pRF.pRF_plan_data,
                   sco.anatomy.anatomy_plan_data,
                   sco.analysis.analysis_plan_data,
                   sco.util.export_plan_data,
                   {'default_options': provide_default_options}]
     for (k,v) in six.iteritems(pd)})

sco_plan      = _pimms.plan(sco_plan_data)
