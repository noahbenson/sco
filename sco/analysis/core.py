####################################################################################################
# sco/analysis/core.py
# Core implementation of the analysis module of the sco library
# by Noah C. Benson

import numpy as np
import scipy as sp, scipy.spatial as sps
import pyrsistent as pyr
import nibabel, nibabel.freesurfer.mghformat as fsmgh
import os, sys, pimms
from sco.util import units, import_mri, nearest_indices, apply_affine

@pimms.calc('measurements', 'measurement_indices',
            'measurement_coordinates', 'measurement_hemispheres',
            cache=True)
def import_measurements(measurements_filename=None):
    '''
    import_measurements is a calculator that imports measured data from a given filename or pair
    of filenames (in the case of surface modalities), and converts them into a matrix of
    measurement values that is the same size os the matrix 'predictions' produced by the sco plan.

    Required afferent values:
      @ measurements_filename Must be either the filename of a volume of measurements or a tuple of
        (lh_filename, rh_filename) if surface files are provided.

    Provided efferent values:
      @ measurements Will be an (n x m) matrix of the measured values whose rows correspond to the
        anatomical ids and whose columns correspond to the images.
      @ measurement_indices Will be a list, one per measurement voxel/vertex whose data appears in
        the measurements matrix, of the voxel-index triple (i,j,k) or the vertex id of each
        measurement; in the latter case, right-hemisphere ids will overlap with left-hemisphere ids,
        and the measurement_hemispheres value must be used to distinguish them.
      @ measurement_coordinates Will be a list of coordinates, one per element of the measurements
        matrix; if the imported measurements were from surface files, then this is None. Note that
        when the measurements filename refers to a NifTI file, this *always* uses the qform affine
        transform to produce coordinates.
      @ measurement_hemispheres Will be a list whose values are all either +1 or -1, one per row of
        the measurements matrix if the measurements that are imported come from surface files; if
        they do not, then this will be None. For surface files, +1 and -1 indicate LH and RH,
        respectively.
    '''
    if measurements_filename is None: return (None, None, None, None)
    meas = None
    idcs = None
    crds = None
    hems = None
    if len(measurements_filename) == 2:
        if pimms.is_map(measurements_filename):
            try:
                measurements_filename = (measurements_filename['lh'], measurements_filename['rh'])
            except:
                measurements_filename = (measurements_filename['LH'], measurements_filename['RH'])
        meas = [None,None]
        idcs = [None,None]
        hems = [None,None]
        for (fnm, hsgn, hidx) in zip(measurements_filename, [1,-1], [0,1]):
            vol = import_mri(fnm)
            vol = vol if len(vol) == n else vol.T
            if len(vol.shape) != 2:
                raise ValueError('measurement surf file %s must have 2 non-unit dimensions' % fnm)
            vstd = np.std(vol, axis=1)
            ii = np.where(np.isfinite(vstd) * ~np.isclose(vstd, 0))[0]
            if len(ii) == 0:
                raise ValueError('measurement surf file %s contained no valid rows' % fnm)
            meas[hidx] = vol
            idcs[hidx] = ii
            hems[hidx] = np.full(len(vol), hsgn, dtype=np.int)
        if not np.array_equal(hsz[0][1:], hsz[1][1:]):
            raise ValueError('(LH,RH) measurements dims must be the same: (%d, %d)' % tuple(hsz))
        (meas, idcs, hems) = [np.concatenate(x, axis=0) for x in [meas, idcs, hems]]
    else:
        img = import_mri(measurements_filename, 'object')
        vol = img.dataobj.get_unscaled()
        if len(vol.shape) != 4: raise ValueError('measurement volume files must have 4 dimensions')
        h = img.header
        tx = img.affine if isinstance(img, fsmgh.MGHImage) else h.get_qform()
        # we need to find the valid voxels; these are the ones that have non-zero variance and
        # that contain no NaNs or infinite values
        vstd = np.std(vol, axis=-1)
        idcs = np.where(np.isfinite(vstd) * ~np.isclose(vstd, 0))
        meas = vol[idcs]
        idcs = np.asarray(idcs, dtype=np.int).T
        crds = apply_affine(tx, np.asarray(idcs, dtype=np.float))
    for x in [meas, idcs, crds, hems]:
        if x is not None: x.setflags(write=False)
    return (meas, idcs, crds, hems)

@pimms.calc('measurement_per_prediction', 'prediction_per_measurement', 'corresponding_indices',
            cache=True)
def calc_correspondence_maps(coordinates, cortex_indices, hemispheres, modality,
                             measurement_coordinates, measurement_indices,
                             measurement_hemispheres):
    '''
    calc_correspondence_maps is a calculator that produces the maps of the indices of the
    measurement rows that go with each prediction row and vice versa; the efferent values are two
    immutable dictionary whose keys are prediction indices and whose values are measurement indices
    (or vice versa); prediction/measurement indices that do not have a matched
    measurement/prediction index do not appear in the maps.

    Provided efferent values:
      @ measurement_per_prediction Will be a map whose keys are indices into the rows of the
        prediction matrix and whose values are the matching rows in the measurement matrix.
      @ prediction_per_measurement Will be a map whose keys are indices into the rows of the
        measurement matrix and whose values are the matching rows in the prediction matrix.
      @ corresponding_indices Will be a 2 x n matrix where n is the number of corresponding voxels
        or vertices in the measurements and predictions; the first row is the indices into the
        prediction matrix and the second row is the matching indices in the measurement matrix.
    '''
    if measurement_coordinates is None:
        if measurement_indices is None:
            return (None, None, None)
        elif modality == 'surface':
            # This means they're both surfaces; we just get vertex overlaps
            ((mlmap, mrmap), (plmap, prmap)) = [
                [{i:k for (k,(i,h)) in enumerate(zip(idcs,hems)) if h == v} for v in [-1,1]]
                for (idcs,hemis) in [(measurement_indices, measurement_hemispheres),
                                     (cortex_indices,      hemispheres)]]
            (lolap, rolap) = [
                set(np.intersect1d(measurement_indices[np.where(measurement_hemispheres == v)[0]],
                                   cortex_indices[     np.where(hemispheres             == v)[0]]))
                for (hi,hv) in [(0,1), (1,-1)]]
            (mpp, ppm) = [pyr.pmap({kmap[i]:vmap[i]
                                    for (kmap,vmap,olap) in [kmaps, vmaps]
                                    for i in olap})
                          for (kmaps, vmaps) in [((mlmap,plmap,lolap),(mrmap,prmap,rolap)),
                                                 ((mrmap,prmap,rolap),(mlmap,plmap,lolap))]]
        else:
            # This cannot hold: we don't know where the measured surface vertices are and we don't
            # have voxel-to-surface data for the predictions
            warnings.warn('predicted volumes and measured surfaces cannot be matched together')
            return (None, None, None)
    else:
        # This means the measurements are voxels and have coordinates; it doesn't matter if the
        # predictions are volume or surface-based--they have coordinates either way
        pidcs = nearest_indices(measurement_coordinates, coordinates)
        midcs = nearest_indices(coordinates, measurement_coordinates)
        (mpp, ppm) = tuple([pyr.pmap({k:v for (k,v) in enumerate(ii)}) for ii in [pidcs, midcs]])
    # Okay, we have the basic correspondence, but we want to know if there are voxels outside of the
    # relevant space; i.e., if the measured voxels fill the whole brain and the prediction is only
    # for v1-v3, we want to trim out the measurement voxels outside of v1-v3.
    # To do this, we trim out voxels from each map that don't appear in the other; i.e., if a
    # measured voxel matches a predicted voxel but that predicted voxel is closer to another
    # measured voxel, we don't use it. This will result in a single minimal mapping, which we store
    # in a 2 x n array of [prediction_indices, measurement_indices]
    cind = np.asarray([(i,j) for (i,j) in mpp.iteritems() if j in ppm and ppm[j] == i],
                      dtype=np.int).T
    return (mpp, ppm, cind)

@pimms.calc('measurement_labels', 'measurement_pRFs', 'measurement_polar_angles',
            'measurement_eccentricities', cache=True)
def calc_correspondence_data(corresponding_indices, labels, pRFs, polar_angles, eccentricities,
                             measurement_indices):
    '''
    calc_correspondence_data is a calculator that yields measurement data translations for various
    data that exist for predictions (such as retintopy).    
    '''
    if corresponding_indices is None:
        return (None, None, None, None)
    n = len(measurement_indices)
    (pidcs, midcs) = corresponding_indices
    mlabs = np.full(n, 0, dtype=np.int32)
    mprfs = np.full(n, None, dtype=np.object)
    mangs = np.full(n, np.nan, dtype=np.float32)
    meccs = np.full(n, np.nan, dtype=np.float32)
    mlabs[midcs] = labels[pidcs]
    mprfs[midcs] = pRFs[pidcs]
    mangs[midcs] = polar_angles[pidcs]
    meccs[midcs] = eccentricities[pidcs]
    for m in [mlabs, mprfs, mangs, meccs]:
        m.setflags(write=False)
    return (mlabs, mprfs, mangs, meccs)

@pimms.calc('prediction_analysis', 'prediction_analysis_labels', memoize=True)
def calc_prediction_analysis(prediction, measurements, labels, hemispheres, pRFs,
                             corresponding_indices):
    '''
    calc_prediction_analysis is a calculator that takes predictions from the sco model and a
    measurement dataset and produces an analysis of the prediction accuracy, ready to be plotted or
    exported.

    Required afferent values:
      * prediction (from sco.contrast)
      * measurements (from sco.analysis)
      * labels, hemispheres (from sco.anatomy)
      * pRFs (from sco.pRF)
      * corresponding_indices

    Provided efferent values:
      * prediction_analysis: a mapping of analysis results, broken down by label and hemisphere
        as well as eccentricity and polar angle. These analyses are in many cases itables that can be
        easily exported as CSV files.
    '''
    if measurements is None: return (None, None)
    (pidcs,midcs) = corresponding_indices
    (prediction,labels,hemispheres,pRFs) = [x[pidcs] for x in (prediction,labels,hemispheres,pRFs)]
    measurements = measurements[midcs]
    if not np.array_equal(prediction.shape, measurements.shape):
        raise ValueError('prediction and measurement sizes are not the same')
    # For starters, we just get the correlations between predictions and measurements:
    r = np.zeros(prediction.shape[0], dtype=np.float)
    for (i,p,g) in zip(range(len(r)), prediction, measurements):
        try:    r[i] = np.corrcoef(p,g)[0,1]
        except: r[i] = np.nan
    rok = np.where(np.isfinite(r))[0]
    # Okay, lets get some indexes ready; we're going to continue to build these up then at the end
    # we'll calculate correlations over all of them
    lbls = {pyr.m(label=lbl):np.intersect1d(rok, np.where(labels == lbl)[0]) for lbl in np.unique(labels)}
    unq_ls = lbls.keys()
    lbls[pyr.m(hemi='lh')] = np.intersect1d(rok, np.where(hemispheres == 1)[0])
    lbls[pyr.m(hemi='rh')] = np.intersect1d(rok, np.where(hemispheres == -1)[0])
    for l in unq_ls:
        for h in ['lh', 'rh']:
            lbls[l.set('hemi',h)] = np.intersect1d(lbls[l], lbls[pyr.m(hemi=h)])
    general_lbls = lbls.keys() # save these for later
    # Alright, next we want to calc our own eccentricities and polar angles
    centers = np.asarray([p.center.to(units.degree).m for p in pRFs])
    ecc  = np.sqrt(np.sum(centers**2, axis=1))
    angm = np.arctan2(centers[:,1], centers[:,0])
    angp = 90.0 - 180.0/np.pi * angm
    angp[angp > 180] -= 360.0
    rad  = np.asarray([p.radius.to(units.degree).m for p in pRFs])
    # Okay, now we can do some interesting comparisons; let's sort by polar angle and eccentricity:
    ecc_bins = int(np.median([10, 100, int(np.round(np.power(len(ecc), 0.375)))]))
    c = 100.0 / (ecc_bins + 1)
    ecc_bin_walls = np.percentile(ecc, c*np.asarray(range(ecc_bins + 1), dtype=np.float))
    ecc_bin_walls = [(mn,mx) for (mn,mx) in zip(ecc_bin_walls[:-1], ecc_bin_walls[1:])]
    ecc_bin_idcs = [np.where((ecc >= mn) & (ecc <= mx))[0] for (mn,mx) in ecc_bin_walls]
    ecc_bin_means = np.asarray([np.mean(ecc[ii]) for ii in ecc_bin_idcs])
    for l0 in general_lbls:
        for (mu,ii) in zip(ecc_bin_means, ecc_bin_idcs):
            lbls[l0.set('eccentricity', mu)] = np.intersect1d(lbls[l0], ii)
    for (mu,ii) in zip(ecc_bin_means, ecc_bin_idcs):
        lbls[pyr.m(eccentricity=mu)] = np.intersect1d(lbls[l0], ii)
    ang_bins = int(np.median([10, 100, int(np.round(np.power(len(angp), 0.375)))]))
    c = 100.0 / (ang_bins + 1)
    ang_bin_walls = np.percentile(angp, c*np.asarray(range(ang_bins + 1), dtype=np.float))
    ang_bin_walls = [(mn,mx) for (mn,mx) in zip(ang_bin_walls[:-1], ang_bin_walls[1:])]
    ang_bin_idcs = [np.where((angp >= mn) & (angp <= mx))[0] for (mn,mx) in ang_bin_walls]
    ang_bin_means = np.asarray([np.mean(angp[ii]) for ii in ang_bin_idcs])
    for l0 in general_lbls:
        for (mu,ii) in zip(ang_bin_means, ang_bin_idcs):
            lbls[l0.set('polar_angle', mu)] = np.intersect1d(lbls[l0], ii)
    for (mu,ii) in zip(ang_bin_means, ang_bin_idcs):
        lbls[pyr.m(polar_angle=mu)] = np.intersect1d(lbls[l0], ii)
    # Okay, now we have the labels divide up; let's look at correlations of vertices across images;
    res = {}
    for (lbl, idcs) in lbls.iteritems():
        if len(idcs) == 0: continue
        # first, calculate mean responses across the area
        mu_pred = np.mean(prediction[idcs, :],   axis=0)
        mu_trth = np.mean(measurements[idcs, :], axis=0)
        # store the correlation of these arrays
        try:    res[lbl] = np.corrcoef(mu_pred, mu_trth)[0,1]
        except: res[lbl] = np.nan
    # That's it!
    return (pyr.pmap(res), pyr.pmap({k:pyr.pvector(v) for (k,v) in lbls.iteritems()}))
    
