####################################################################################################
# sco/util/__init__.py
# Utilities useful in work related to the sco predictions.
# By Noah C. Benson

import pyrsistent as _pyr
import pimms      as _pimms

from .plot import (cortical_image, corrcoef_image, report_image)
from .io   import (export_predictions, export_analysis, export_report_images, calc_exported_files,
                   export_vega, require_exports)
# The unit registry that we use
units = _pimms.units

def lookup_labels(labels, data_by_labels, **kwargs):
    '''
    sco.util.lookup_labels(labels, data_by_labels) yields a list the same size as labels in which
      each element i of the list is equal to data_by_labels[labels[i]].
    
    The option null may additionally be passed to lookup_labels; if null is given, then whenever a
    label value from data is not found in labels, it is instead given the value null; if null is not
    given, then an error is raised in this situation.

    If the data_by_labels given is a string, then lookup_labels attempts to use the value
    global_lookup(data_by_labels) in its place.

    The lookup_labels function expects the labels to be integer or numerical values.
    '''
    res = None
    null = None
    raise_q = True
    if isinstance(data_by_labels, basestring): data_by_labels = global_lookup(data_by_labels)
    if 'null' in kwargs:
        null = kwargs['null']
        raise_q = False
    if len(kwargs) > 1 or (len(kwargs) > 0 and 'null' not in kwargs):
        raise ValueError('Unexpected option given to lookup_labels; only null is accepted')
    if raise_q:
        try:
            res = [data_by_labels[lbl] for lbl in labels]
        except:
            raise ValueError('Not all labels found by lookup_labels and no null given')
    else:
        res = [data_by_labels[lbl] if lbl in data_by_labels else null for lbl in labels]
    return _pyr.pvector(res)

export_plan_data = _pyr.m(export_predictions     = export_predictions,
                          export_analysis        = export_analysis,
                          export_report_images   = export_report_images,
                          export_vega            = export_vega,
                          exported_files         = calc_exported_files)
export_plan = _pimms.plan(export_plan_data)


# Some additional handy functions
def import_mri(filename, feature='data'):
    '''
    import_mri(filename) yields a numpy array of the data imported from the given filename. The
      filename argument must be a string giving the name of a NifTI (*.nii or *.nii.gz) or MGH
      (*.mgh, *.mgz) file. The data is squeezed prior to being returned.
    import_mri(filename, feature) yields a specific feature of the object imported from filename;
      these features are given below.

    Features:
      * 'data'    equivalent to import_mri(filename).
      * 'header'  yields the header of the nibabel object representing the volume.
      * 'object'  yields the nibabel object representing the volume.
      * 'affine'  yields the affine transform of the given volume file; for an MGH file this is
                  object.affine; for a NifTI file this is object.get_best_affine().
      * 'qform'   yields the qform matrix for a NifTI file; raises an exception for an MGH file.
      * 'sform'   yields the qform matrix for a NifTI file; raises an exception for an MGH file.
      * 'vox2ras' yields object.get_vox2ras_tkr() for an MGH file; raises an exception for an MGH
                  file.
      * 'rawdata' identical to 'data' except that the data is not squeezed.
    '''
    import nibabel as nib, nibabel.freesurfer.mghformat as mgh
    if feature is None: feature = 'object'
    if not isinstance(feature, basestring): raise ValueError('feature must be a string or None')
    if feature == 'all': feature = 'object'
    # go ahead and get the file
    try:    obj = nib.load(filename)
    except: obj = mgh.load(filename)
    # okay, now interpret the data
    feature = feature.lower()
    if   feature == 'object':  return obj
    elif feature == 'header':  return obj.header
    elif feature == 'data':    return np.squeeze(obj.dataobj.get_unscaled())
    elif feature == 'rawdata': return obj.dataobj.get_unscaled()
    elif feature == 'affine':
        return obj.affine if isinstance(obj, mgh.MGHImage) else obj.header.get_best_affine()
    elif feature == 'qform':
        if isinstance(obj, mgh.MGHImage):
            raise ValueError('MGH object do not have qforms')
        return obj.header.get_qform()
    elif feature == 'sform':
        if isinstance(obj, mgh.MGHImage):
            raise ValueError('MGH object do not have sforms')
        return obj.header.get_sform()
    elif feature == 'vox2ras':
        if not isinstance(obj. mgh.MGHImage):
            raise ValueError('NifTI files do not have vox2ras matrices')
        return obj.header.get_vox2ras_tkr()
    else:
        raise ValueError('unrecognized feature: %s' % feature)
    
def apply_affine(affine_matrix, points):
    '''
    apply_affine(affine_matrix, points) yields the n x 3 matrix that results from applying the
      affine transform given (n x 3) matrix of points. The affine transform must be stored as
      either a 4 x 4 matrix or a tuple of (mtx, x0) where mtx is a 3x3 matrix and x0 is a 3d
      vector giving the displacement.
    '''
    import numpy as np
    if len(points.shape) < 2: raise ValueError('points must be a matrix')
    if points.shape[1] == 3: points = points.T
    if isinstance(affine_matrix, tuple):
        return np.dot(affine_matrix[0], points).T + affine_matrix[1]
    else:
        ones = np.ones((1, points.shape[1]))
        return np.dot(affine_matrix, np.concatenate((points, ones)))[:-1].T

def nearest_indices(database_coords, search_coords):
    '''
    nearest_indices(database_coords, search_coords) yields a numpy array of the indices (one index
      per search coordinate) of the database coordinate closest to the corresponding search
      coordinate.
    This function is equivalent to the following:
      scipy.spatial.cKDTree(database_coords).query(search_coords, 1)[1]
    '''
    import scipy
    return scipy.spatial.cKDTree(database_coords).query(search_coords, 1)[1]

def global_lookup(s):
    '''
    global_lookup(s) yields the value referenecd by the string s; this cnverts between, for example,
      a string like 'sco.impl.benson17.divnorm_heeger91' into the named function, if it can be
      loaded; otherwise an error is raised.
    '''
    import importlib
    (mname, fname) = s.rsplit('.', 1)
    return getattr(importlib.import_module(mname), fname)
