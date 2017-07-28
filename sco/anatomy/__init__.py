####################################################################################################
# anatomy/__init__.py
# The anatomy module of the standard cortical observer.
# By Noah C. Benson

'''
The sco.anatomy module of the standard cortical observer library is responsible for importing and
interpreting anatomical data from a subject and providing (and tracking) a list of parameters for
each voxel. Additionally, the anatomy module includes utilities for writing out results as a
FreeSurfer volume.

The anatomy module defines an abstract base class, AnatomyBase, whose abstract methods define the
interface for the module. When constructing an SCO pipeline, an AnatomyBase object is required,
and one may be obtained from the StandardAnatomy class or from a custom class that overloads the
AnatomyBase class.
'''

import pyrsistent as _pyr
import pimms as _pimms
from .core  import (import_freesurfer_subject,
                    import_freesurfer_affine,
                    import_benson14_from_freesurfer,
                    import_retinotopy_data_files,
                    calc_prediction_coordinates,
                    export_predicted_responses)

# Make a function that's ready to be used as a module
anatomy_plan_data = _pyr.m(import_subject    = import_freesurfer_subject,
                           import_affine_tx  = import_freesurfer_affine,
                           import_retinotopy = import_benson14_from_freesurfer,
                           calc_coordinates  = calc_prediction_coordinates)

anatomy_plan = _pimms.plan(anatomy_plan_data)
