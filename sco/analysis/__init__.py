####################################################################################################
# sco/analysis/__init__.py
# Definition code for the analysis module of the sco library
# by Noah C. Benson

import pimms as _pimms
import pyrsistent as _pyr
from .core import (import_measurements, calc_prediction_analysis, calc_correspondence_maps)

analysis_plan_data = _pyr.m(import_measurements = import_measurements,
                            calc_correspondence = calc_correspondence_maps,
                            prediction_analysis = calc_prediction_analysis)

analysis_plan = _pimms.plan(analysis_plan_data)

