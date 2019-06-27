####################################################################################################
# contrast/__init__.py
# The second-order contrast module of the standard cortical observer.
# By Noah C. Benson

'''
The sco.contrast module of the standard cortical observer library is responsible for calculating the
first- and second-order contrast present in the normalized stimulus image array.
'''

import pyrsistent as _pyr
import pimms as _pimms
from .core import (calc_contrast_constants,        calc_compressive_constants, 
                   calc_gabor_spatial_frequencies, calc_oriented_contrast_images,
                   calc_divisive_normalization,    calc_pRF_contrasts,
                   calc_gains,                     calc_compressive_nonlinearity)

contrast_plan_data = _pyr.m(contrast_constants        = calc_contrast_constants,
                            compressive_constants     = calc_compressive_constants,
                            gabor_spatial_frequencies = calc_gabor_spatial_frequencies,
                            oriented_contrast_images  = calc_oriented_contrast_images,
                            divisive_normalization    = calc_divisive_normalization,
                            pRF_contrasts             = calc_pRF_contrasts,
                            gains                     = calc_gains,
                            compressive_nonlinearity  = calc_compressive_nonlinearity)

contrast_plan = _pimms.plan(contrast_plan_data)

