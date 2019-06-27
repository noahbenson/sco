####################################################################################################
# sco/impl/benson17/__init__.py
# Implementation details of the SCO specific to the Benson et al. 2017 publications.
# by Noah C. Benson

'''
The sco.impl.testing namespace contains default values and special model constructions for all a
testing/debugging version of the SCO model. This testing model is designed to create pRFs to fill an
image (thus require no explicit subject) and to generate specific stimulus files.
'''

from .core import (sco_plan_data, sco_plan,
                   noise_pattern_stimulus, luminance_grating, test_grating, test_stimuli,
                   calc_image_retinotopy, calc_stimulus, provide_default_options)
