####################################################################################################
# sco/util/io.py
# Import/export utilities for the sco project
# by Noah C. Benson

import numpy as np
import pyrsistent as pyr
import json
import os, pimms, sys
import nibabel, nibabel.freesurfer.mghformat as fsmgh

from .plot import (cortical_image, corrcoef_image, report_image)

def _sco_init_outputs(output_directory, create_directories, output_prefix, output_suffix):
    if not os.path.exists(output_directory) or not os.path.isdir(output_directory):
        if create_directories:
            os.mkdirs(output_directory)
        if not (create_directories
                and os.path.exists(output_directory) and os.path.isdir(output_directory)):
            raise ValueError('Output directory does not exist')
    output_prefix = '' if output_prefix is None else output_prefix
    output_suffix = '' if output_suffix is None else output_suffix
    return (output_prefix, output_suffix)

def export_anatomical_data(data, sub, anat_ids, hemis, name, output_dir,
                           modality=None, create_dir=False, prefix='', suffix='',
                           null=np.nan, dtype=None):
    '''
    export_anatomical_data(data, sub, cortex_indices, hemis, name, output_dir) exports the
       prediction data in the given data vector or matrix, out to the given directory in a set of
       files appropriate to the modality; for surfaces, this is an lh.<name>.nii.gz and
       rh.<name>.nii.gz file; for volumes, this is just a <name>.nii.gz file.

    Parameters:
      * data must be a vector of data or a matrix of data whose rows match up to the anatomical ids
      * cortex_indices must be the voxel indices or vertex ids of the data rows
      * hemis must be a list of 1 or -1 values to indicate LH and RH respectively
      * name should be a name for the file, used as a part of the filename such as lh.<name>.nii.gz
      * output_dir must be the name of the output directory
      * freesurfer_subject: the freesurfer subject object

    Options:
      * create_dir (default: False) if True, will create the directory if it does not exist;
        otherwise raises an exception when the directory does not exist.
      * prefix (default: '') may be a string that is prepended to filenames; prepends are performed
        before other prefixes such as 'lh.<prefix><name><suffix>.nii.gz'.
      * suffix (default: '') may be a string that is appended the the filename; appends are
        performed before the file extension is appended, such as in
        'lh.<prefix><name><suffix>.nii.gz'.
      * modality (default: None) may be specified explicitly as 'surface' or 'volume', but will
        auto-detect the modality based on the cortex_indices if None is given
      * null (default: nan) specifies the null value that should be written for values not in the
        cortex_indices; if this is nan and the dtype is an integer value, then 0 is used instead
      * dtype (default: None) specifies explicitly the dtype of the exported array; if None, the
        value np.asarray(data).dtype is used
    '''
    anat_ids = np.asarray(anat_ids)
    if len(anat_ids.shape) == 2 and (anat_ids.shape[0] == 1 or anat_ids.shape[1] == 1):
        anat_ids = anat_ids.flatten()
    if modality is None:
        modality = 'volume' if len(anat_ids.shape) == 2 else 'surface'
    (prefix, suffix) = _sco_init_outputs(output_dir, create_dir, prefix, suffix)
    make_fname = lambda p,ext: os.path.join(output_dir, p + prefix + name + suffix + '.' + ext)
    modality = modality.lower()
    data = np.asarray(data)
    data = np.asarray([data]).T if len(data.shape) == 1 else data
    if len(data) != len(anat_ids):
        raise ValueError('anatomical id (%d) and data (%d) sizesmust match' % (
            len(anat_ids), len(data)))
    if dtype is None: dtype = data.dtype
    if np.issubdtype(dtype, np.dtype(int).type) and not np.isfinite(null): null = 0
    affine = sub.mgh_images['lh.ribbon'].affine
    file_names = []
    if modality == 'surface':
        for (hname, hid) in [('lh', 1), ('rh', -1)]:
            hobj = getattr(sub, hname.upper())
            hidcs = np.where(hemis == hid)[0]
            n = hobj.vertex_count
            if np.issubdtype(dtype, np.float64): dtype = np.float32
            if data.shape[1] == 1:
                vol = np.full((1,1,n), null, dtype=dtype)
                vol[0,0,anat_ids[hidcs]] = data[hidcs,0]
            else:
                vol = np.full((1,1,n,data.shape[1]), null, dtype=dtype)
                vol[0,0,anat_ids[hidcs],:] = data[hidcs,:]
            #img = nibabel.Nifti2Image(vol, np.eye(4))
            #fnm = make_fname(hname + '.', 'nii.gz')
            img = fsmgh.MGHImage(vol, np.eye(4))
            fnm = make_fname(hname + '.', 'mgz')
            img.to_filename(fnm)
            file_names.append(fnm)
    elif modality == 'volume':
        vol = np.full(sub.mgh_images['lh.ribbon'].shape + (data.shape[1],), null, dtype=dtype)
        for ((i,j,k),row) in zip(anat_ids, data): vol[i,j,k,:] = row
        img = nibabel.Nifti1Image(vol, affine)
        fnm = make_fname('', 'nii.gz')
        img.to_filename(fnm)
        file_names.append(fnm)
    else:
        raise ValueError('unrecognized modality: %s' % modality)
    return file_names

@pimms.calc('exported_predictions_filenames')
def export_predictions(prediction, cortex_indices, modality, hemispheres, freesurfer_subject,
                       labels, image_names, output_directory='.',
                       create_directories=False, output_prefix='', output_suffix=''):
    '''
    export_predictions is a calc that exports the prediction data in from the sco calculation, which
       must come from sco.sco_plan(...) or similar; at the least it must contain the data documented
       below. The return value is a set of filenames exported.

    Required afferent values:
      * prediction:         the prediction matrix
      * cortex_indices:     the voxel indices or vertex ids
      * image_names:        the list of image names in order
      * modality:           'surface' or 'volume'
      * labels:             the anatomical labels
      * freesurfer_subject: the freesurfer subject object
      @ output_directory (default: '.') the directory to which to write the results; if None, then
        uses the current directory (.).

    Options:
      * create_directories (default: False) if True, will create the directory if it does not exist;
        otherwise raises an exception when the directory does not exist.
      * output_prefix (default: '') may be a string that is prepended to filenames; prepends are
        performed before other prefixes such as 'lh.<prefix><name><suffix>.nii.gz'.
      * output_suffix (default: '') may be a string that is appended the the filename; appends are
        performed before the file extension is appended, such as in
        'lh.<prefix><name><suffix>.nii.gz'.

    Provided efferent values:
      @ exported_predictions_filenames Will be a list of the prediction files exported by
        export_predictions.
    '''
    (output_prefix, output_suffix) = _sco_init_outputs(output_directory, create_directories,
                                                       output_prefix, output_suffix)
    # output the raw predictions themselves
    fnms = export_anatomical_data(prediction, freesurfer_subject, cortex_indices, hemispheres,
                                  'prediction', output_directory,
                                  create_dir=create_directories, modality=modality,
                                  prefix=output_prefix, suffix=output_suffix)
    # also the image name list
    imnms_fnm = os.path.join(output_directory, output_prefix + 'images' + output_suffix + '.txt')
    with open(imnms_fnm, 'w') as f:
        for imnm in image_names:
            f.write(imnm + '\n')
    fnms.append(imnms_fnm)
    # that's all this calculator does
    return pyr.pvector(fnms)

@pimms.calc('exported_analysis_filenames')
def export_analysis(prediction_analysis, prediction_analysis_labels, output_directory='.',
                    create_directories=False, output_prefix='', output_suffix=''):
    '''
    export_evaluations is a calculator that exports the evaluation data in the calc plan, which
       must come from sco.sco_plan(...) or similar; at the least it must contain the data documented
       below. The return value is a set of filenames exported.

    Required afferent values:
      * prediction_analysis and prediction_analysis_labels: the analysis of the predicted versus
        measurements data (from sco.analysis)
      * output_directory, the directory to which to write the results

    Options:
      * create_directories (default: False) if True, will create the directory if it does not exist;
        otherwise raises an exception when the directory does not exist.
      * output_prefix (default: '') may be a string that is prepended to filenames; prepends are
        performed before other prefixes such as 'lh.<prefix><name><suffix>.nii.gz'.
      * output_suffix (default: '') may be a string that is appended the the filename; appends are
        performed before the file extension is appended, such as in
        'lh.<prefix><name><suffix>.nii.gz'.

    Provided efferent values:
      @ exported_analysis_filenames Will be a list of filenames exported by export_analysis.
    '''
    if prediction_analysis is None: return None
    # We basically just make a big CSV data-table:
    (output_prefix, output_suffix) = _sco_init_outputs(output_directory, create_directories,
                                                       output_prefix, output_suffix)
    # Write out the label-group correlation data
    filename = os.path.join(output_directory, output_prefix + 'analysis' + output_suffix + '.csv')
    headers = sorted(list(set([k for m in prediction_analysis.iterkeys() for k in m.iterkeys()])))
    header = ''
    fmt = ''
    for h in headers:
        header += ',' + h
        fmt += ',%s'
    header = '#' + header[1:] + ',n,correlation\n'
    fmt = fmt[1:] + ',%s,%s\n'
    with open(filename, 'w') as f:
        f.write(header)
        for (lbl, rval) in prediction_analysis.iteritems():
            n = len(prediction_analysis_labels[lbl])
            tup = tuple([lbl[k] if k in lbl else 'all' for k in headers])
            tup = tup + (n, rval)
            f.write(fmt % tup)
    return pyr.v(filename)

@pimms.calc('exported_report_filenames')
def export_report_images(labels, pRFs, max_eccentricity,
                         prediction_analysis, measurements, output_directory='.',
                         create_directories=False, output_prefix='', output_suffix=''):
    '''
    export_report_images is a calculator that takes a prediction analysis
      (see sco.analysis.calc_prediction_analysis) and exports a set of images to the output
      directory that report on the accuracy of the model predictions.

    Note that this calculator does nothing and simply yields None if the measurements or
    prediction_analysis values are not found; these have default values.

    Required afferent values:
      * output_directory, the directory to which to write the results

    Options:
      * prediction_analysis and prediction_analysis_labels: the analysis of the predicted versus
        measured data (from sco.analysis)
      * create_directories (default: False) if True, will create the directory if it does not exist;
        otherwise raises an exception when the directory does not exist.
      * output_prefix (default: '') may be a string that is prepended to filenames; prepends are
        performed before other prefixes such as 'lh.<prefix><name><suffix>.nii.gz'.
      * output_suffix (default: '') may be a string that is appended the the filename; appends are
        performed before the file extension is appended, such as in
        'lh.<prefix><name><suffix>.nii.gz'.

    Provided efferent values:
      @ exported_report_filenames Will be a list of filenames of analysis images exported.
    '''
    try:
        import matplotlib.pyplot as plt
    except:
        raise RuntimeError('Could not import matplotlib.pyplot; matplotlib may not be installed')

    
    if prediction_analysis is None or measurements is None: return None
    (output_prefix, output_suffix) = _sco_init_outputs(output_directory, create_directories,
                                                       output_prefix, output_suffix)
    fnm = os.path.join(output_directory, output_prefix + 'accuracy' + output_suffix + '.pdf')
    f = report_image(prediction_analysis)
    f.savefig(fnm)
    fnms = [fnm]
    plt.close(f)
    for l in np.unique(labels):
        fnm = os.path.join(output_directory,
                           output_prefix + ('v%dcorr' % l) + output_suffix + '.png')
        fnms.append(fnm)
        f = corrcoef_image(prediction_analysis, measurements, labels, pRFs, max_eccentricity,
                           visual_area=l)
        f.savefig(fnm)
        plt.close(f)
    return pyr.pvector(fnms)

@pimms.calc('exported_vega')
def export_vega(prediction_analysis, prediction_analysis_labels,
                prediction, measurements, corresponding_indices, output_directory='.',
                create_directories=False, output_prefix='', output_suffix=''):
    '''
    export_vega is a pimms calculation that takes mostly analyzed data and exports a vega-lite file
    rendering a histogram of the accuracies.
    '''
    (output_prefix,output_suffix) = ['' if x is None else x for x in (output_prefix,output_suffix)]
    if measurements is None: return (None, None)
    (pidcs,midcs) = corresponding_indices
    idcs = prediction_analysis_labels[pyr.m(hemi='lh',label=1)]
    pidcs = pidcs[idcs]
    midcs = midcs[idcs]
    prediction   = prediction[pidcs]
    measurements = measurements[midcs]
    rs = np.asarray([np.corrcoef(p,m)[0,1] for (p,m) in zip(prediction, measurements)])
    json_spec_fnm = os.path.join(output_directory,
                            output_prefix + 'vega-corthist-spec' + output_suffix + '.json')
    json_fnm  = os.path.join(output_directory,
                            output_prefix + 'vega-corthist' + output_suffix + '.json')
    vega_spec = '''
      {"$schema": "https://vega.github.io/schema/vega-lite/v2.json",
       "data": {"url": "%s"},
       "mark": "bar",
       "encoding": {
         "x": {
           "bin": {"maxbins": 10},
           "field": "correlation",
           "type": "quantitative"},
         "y": {
           "aggregate": "count",
           "type": "quantitative"}}}
      '''
    with open(json_spec_fnm, 'w') as f: f.write(vega_spec % json_fnm)
    with open(json_fnm, 'w') as f:
        json.dump([{'pid':p, 'mid':m, 'correlation':r} for (p,m,r) in zip(midcs,pidcs,rs)], f)
    return True

@pimms.calc('exported_files')
def calc_exported_files(exported_report_filenames,
                        exported_analysis_filenames,
                        exported_predictions_filenames,
                        exported_vega):
    '''
    calc_exported_files is a calculation object that simply accumulates the files exported by
      various other functions in the sco.util package (specifically sco.util.io) and stores the list
      of all successfully exported files in the efferent value 'exported_files'.

    Required afferent values:
      @ exported_report_filenames Filename list generally obtained from
        sco.util.io.export_report_images.
      @ exported_predictions_filenames Filename list generally obtained from
        sco.util.io.export_predictions.
      @ exported_analysis_filenames Filename list generally obtained from
        sco.util.export_analysis.
    '''
    flists = [exported_report_filenames,exported_analysis_filenames,exported_predictions_filenames]
    return pyr.pvector([u for flist in flists if flist is not None for u in flist])

@pimms.calc(None)
def require_exports(exported_files):
    '''
    require_exports is a calc object that does very little but, when added to a calc plan, requires
      that all exports be completed upon initialization.
    '''
    pass
