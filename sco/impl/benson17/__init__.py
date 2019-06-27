####################################################################################################
# sco/impl/benson17/__init__.py
# Implementation details of the SCO specific to the Benson et al. 2017 publications.
# by Noah C. Benson

'''
The sco.impl.benson17 namespace contains default values for all of the optional parameters used in
the Winawer lab implementation of the SCO model. A function, provide_default_options is also
defined; this may be included in any SCO plan to ensure that the optional parameter values are
available to the downstream calculations if not provided by the user or other calculations.
'''

import pyrsistent                                    as _pyr
import pimms                                         as _pimms
import numpy                                         as _np
import six
from   sco.util                     import units     as _units
from   neuropythy.vision.retinotopy import pRF_data  as _neuropythy_pRF_data

import sco.anatomy
import sco.stimulus
import sco.pRF
import sco.contrast
import sco.analysis
import sco.util


# Parameters Defined by Labels #####################################################################
visual_area_names_by_label = _pyr.pmap({1:'V1', 2:'V2', 3:'V3', 4:'hV4'})
visual_area_labels_by_name = _pyr.pmap({v:k for (k,v) in six.iteritems(visual_area_names_by_label)})
pRF_sigma_slopes_by_label_Kay2013      = _pyr.pmap(
    {1:_neuropythy_pRF_data['kay2013']['v1' ]['m'],
     2:_neuropythy_pRF_data['kay2013']['v2' ]['m'],
     3:_neuropythy_pRF_data['kay2013']['v3' ]['m'],
     4:_neuropythy_pRF_data['kay2013']['hv4']['m']})
pRF_sigma_offsets_by_label_Kay2013     = _pyr.pmap(
    {1:_neuropythy_pRF_data['kay2013']['v1' ]['b'],
     2:_neuropythy_pRF_data['kay2013']['v2' ]['b'],
     3:_neuropythy_pRF_data['kay2013']['v3' ]['b'],
     4:_neuropythy_pRF_data['kay2013']['hv4']['b']})
pRF_sigma_slopes_by_label_Wandell2015  = _pyr.pmap(
    {1:_neuropythy_pRF_data['wandell2015']['v1' ]['m'],
     2:_neuropythy_pRF_data['wandell2015']['v2' ]['m'],
     3:_neuropythy_pRF_data['wandell2015']['v3' ]['m'],
     4:_neuropythy_pRF_data['wandell2015']['hv4']['m']})
pRF_sigma_offsets_by_label_Wandell2015 = _pyr.pmap(
    {1:_neuropythy_pRF_data['wandell2015']['v1' ]['b'],
     2:_neuropythy_pRF_data['wandell2015']['v2' ]['b'],
     3:_neuropythy_pRF_data['wandell2015']['v3' ]['b'],
     4:_neuropythy_pRF_data['wandell2015']['hv4']['b']})
contrast_constants_by_label_Kay2013    = _pyr.pmap({1:0.93, 2:0.99, 3:0.99, 4:0.95})
compressive_constants_by_label_Kay2013 = _pyr.pmap({1:0.18, 2:0.13, 3:0.12, 4:0.115})
saturation_constants_by_label_Kay2013  = _pyr.pmap({1:0.50, 2:0.50, 3:0.50, 4:0.50})
divisive_exponents_by_label_Kay2013    = _pyr.pmap({1:1.00, 2:1.00, 3:1.00, 4:1.00})
gains_by_label_Kay2013                 = _pyr.pmap({1:1.00, 2:0.79, 3:0.71, 4:1.23})
#gains_by_label_Kay2013                 = _pyr.pmap({1:1.64, 2:0.66, 3:0.65, 4:1.72})
# Some experimental parameters by labels
ones_by_label  = _pyr.pmap({1:1.0, 2:1.0, 3:1.0, 4:1.0})
zeros_by_label = _pyr.pmap({1:0.0, 2:0.0, 3:0.0, 4:1.0})

# Frequency Sensitivity ############################################################################
def spatial_frequency_sensitivity(prf, cpds):
    '''
    spatial_frequency_sensitivity(prf, cpds) yields a list of weights for the spatial frequencies
      found in the argument cpds (in cycles/degree) for the given prf.
    
    The formula used for this calculation is as follows:
      * r  = pRF eccentricity
      * p0 = preferred spatial period (1/f)
      Then, p0 = 0.13594144711190237 * r + 0.33057571189207119;
      The FWHM in log2 spacing of the Gaussian centered at p0 is ~4.35 (std of ~1.85).
    This formula came from the following work:
      * Broderick WF, Benson NC, Simoncelli E, Winawer J (2018) Mapping spatial frequency
        preferences in the human visual cortex. Vision Sciences Society annual meeting, 2018,
        poster presentation.
    '''
    r  = _pimms.mag(_np.sqrt(prf.center[0]**2 + prf.center[1]**2), 'degree')
    p0 = 0.13594144711190237 * r + 0.33057571189207119
    lf0 = _np.log2(1/p0)
    # weights are log-gaussian distributed around the preferred period
    cpds = _np.log2(_pimms.mag(cpds, 'cycles/degree'))
    ws = _np.exp(-0.5 * ((cpds - lf0)/1.85)**2)
    return ws / _np.sum(ws)

# Default Options ##################################################################################
# The default options are provided here for the SCO
@_pimms.calc('benson17_default_options_used')
def provide_default_options(
        pRF_sigma_slopes_by_label      = 'sco.impl.benson17.pRF_sigma_slopes_by_label_Wandell2015',
        pRF_sigma_offsets_by_label     = 'sco.impl.benson17.pRF_sigma_offsets_by_label_Wandell2015',
        contrast_constants_by_label    = 'sco.impl.benson17.contrast_constants_by_label_Kay2013',
        compressive_constants_by_label = 'sco.impl.benson17.compressive_constants_by_label_Kay2013',
        saturation_constants_by_label  = 'sco.impl.benson17.saturation_constants_by_label_Kay2013',
        gains_by_label                 = 'sco.impl.benson17.gains_by_label_Kay2013',
        max_eccentricity               = 12,
        modality                       = 'surface',
        spatial_frequency_sensitivity_function = 'sco.impl.benson17.spatial_frequency_sensitivity'):
    '''
    provide_default_options is a calculator that optionally accepts values for all parameters for
    which default values are provided in the sco.impl.benson17 package and yields into the calc plan
    these parameter values or the default ones for any not provided.
 
    These options are:
      * pRF_sigma_slopes_by_label (sco.impl.benson17.pRF_sigma_slopes_by_label_Wandell2015)
      * pRF_sigma_offsets_by_label (sco.impl.benson17.pRF_sigma_offsets_by_label_Wandell2015)
      * compressive_constants_by_label (sco.impl.benson17.compressive_constants_by_label_Kay2013)
      * contrast_constants_by_label (sco.impl.benson17.contrast_constants_by_label_Kay2013)
      * modality ('surface')
      * max_eccentricity (12)
      * spatial_frequency_sensitivity_function (sco.impl.benson17.spatial_frequency_sensitivity)
      * saturation_constant_by_label (sco.impl.benson17.saturation_constants_by_label_Kay2013)
      * gains_by_label (sco.impl.benson17.gains_by_label_Kay2013)
      * spatial_frequency_sensitivity_function (sco.impl.benson17.spatial_frequency_sensitivity)
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
                                         {'default_options': provide_default_options}]
                           for (k,v) in six.iteritems(pd)})

sco_plan      = _pimms.plan(sco_plan_data)
