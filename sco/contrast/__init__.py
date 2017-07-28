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
from .core import (ImageArrayContrastView,        ImageArrayContrastFilter,
                   calc_contrast_filter,          calc_contrast_constants,
                   calc_contrast_energies,        calc_compressive_constants,
                   calc_compressive_nonlinearity, calc_pRF_SOC)

contrast_plan_data = _pyr.m(contrast_constants       = calc_contrast_constants,
                            compressive_constants    = calc_compressive_constants,
                            contrast_filter          = calc_contrast_filter,
                            contrast_energies        = calc_contrast_energies,
                            pRF_SOC                  = calc_pRF_SOC,
                            compressive_nonlinearity = calc_compressive_nonlinearity)

contrast_plan = _pimms.plan(contrast_plan_data)

