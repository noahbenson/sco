####################################################################################################
# anatomy/core.py
# The anatomy module of the standard cortical observer; core definitions and checks.
# By Noah C. Benson

import numpy                 as     np
import neuropythy            as     neuro
import neuropythy.commands   as     neurocmd
import nibabel               as     nib
import nibabel.freesurfer    as     fs
import pyrsistent            as     pyr

import pimms, os, sys, warnings

from   ..util                import (lookup_labels, units, import_mri, apply_affine)

@pimms.calc('freesurfer_subject', memoize=True)
def import_freesurfer_subject(subject):
    '''
    import_freesurfer_subject is a calculator that requires a subject id (subject) and yields a
    Neuropythy FreeSurfer subject object, freesurfer_subject.

      @ subject Must be one of (a) the name of a FreeSurfer subject found on the subject path,
        (b) a path to a FreeSurfer subject directory, or (c) a neuropythy FreeSurfer subject
        object.
    '''
    if isinstance(subject, basestring):
        subject = neuro.freesurfer_subject(subject)
    if not isinstance(subject, neuro.Subject):
        raise ValueError('Value given for subject is neither a string nor a neuropythy subject')
    return subject

@pimms.calc('cortex_affine')
def import_freesurfer_affine(freesurfer_subject, modality='surface'):
    '''
    import_freesurfer_affine is a calculation that imports the affine transform associated with a
    freesurfer subject's volumes.

    Required afferent parameters:
      @ freesurfer_subject Must be a valid neuropythy.freesurfer.Subject object.
 
    Optional afferent parameters:
      @ modality May be 'volume' or 'surface' to specify the anatomical modality.

    Exported efferent values:
      @ cortex_affine Will be the affine transformation matrix associated with the given
        freesurfer subject's volume data if the modality is 'volume'; if the modality is 'surface'
        then the affine transformation matrix converts from surface vertex coordinates to the space
        defined by the subject's ribbon's affine (i.e., the volume space, which is distinct from
        the voxel index space).
    '''
    raff = None
    try:
        raff = freesurfer_subject.mgh_images['lh.ribbon'].affine
    except:
        raff = freesurfer_subject.mgh_images['rh.ribbon'].affine
    if modality.lower() == 'volume':
        return raff
    elif modality.lower() == 'surface':
        try:
            tkr = freesurfer_subject.mgh_images['lh.ribbon'].header.get_vox2ras_tkr()
        except:
            tkr = freesurfer_subject.mgh_images['rh.ribbon'].header.get_vox2ras_tkr()
        return np.dot(raff, np.linalg.inv(tkr))
    else:
        raise ValueError('Unknown modality: %s' % modality)

@pimms.calc('polar_angles', 'eccentricities', 'labels', 'hemispheres',
            'cortex_indices', 'cortex_coordinates')
def import_benson14_from_freesurfer(freesurfer_subject, max_eccentricity,
                                    modality='surface', import_filter=None):
    '''
    import_benson14_from_freesurfer is a calculation that imports (or creates then imports) the
    Benson et al. (2014) template of retinotopy for the subject, whose neuropythy.freesurfer
    Subject object must be provided in the parameter freesurfer_subject. The optional parameter
    modality (default: 'volume') may be either 'volume' or 'surface', and determines if the loaded
    modality is volumetric or surface-based.

    Required afferent parameters:
      @ freesurfer_subject Must be a valid neuropythy.freesurfer.Subject object.
 
    Optional afferent parameters:
      @ modality May be 'volume' or 'surface' to specify the anatomical modality.
      @ max_eccentricity May specifies the maximum eccentricity value to use.
      @ import_filter If specified, may give a function that accepts four parameters:
        f(polar_angle, eccentricity, label, hemi); if this function fails to return True for the 
        appropriate values of a particular vertex/voxel, then that vertex/voxel is not included in
        the prediction.

    Provided efferent values:
      @ polar_angles    Polar angle values for each vertex/voxel.
      @ eccentricities  Eccentricity values for each vertex/voxel.
      @ labels          An integer label 1, 2, or 3 for V1, V2, or V3, one per vertex/voxel.
      @ hemispheres     1 if left, -1 if right for all vertex/voxel.
      @ cortex_indices  For vertices, the vertex index (in the appropriate hemisphere) for each;
                        for voxels, the (i,j,k) voxel index for each.
      @ cortex_coordinates For voxels, this is the (i,j,k) voxel index (same as cortex_indices); for
                        surfaces, this is ths (x,y,z) position of each vertex in surface-space.

    Notes:
      * polar_angles are always given such that a negative polar angle indicates a RH value and a
        positive polar angle inidicates a LH value
      * cortex_indices is different for surface and volume modalities
      * labels will always be 1, 2, or 3 indicating V1, V2, or V3

    '''
    max_eccentricity = max_eccentricity.to(units.deg) if pimms.is_quantity(max_eccentricity) else \
                       max_eccentricity*units.deg
    subject = freesurfer_subject
    if modality.lower() == 'volume':
        # make sure there are template volume files that match this subject
        ang = os.path.join(subject.path, 'mri', 'benson14_angle.mgz')
        ecc = os.path.join(subject.path, 'mri', 'benson14_eccen.mgz')
        lab = os.path.join(subject.path, 'mri', 'benson14_varea.mgz')
        if not os.path.exists(ang) or not os.path.exists(ecc) or not os.path.exists(lab):
            # Apply the template first...
            neurocmd.benson14_retinotopy.main(subject.path)
        if not os.path.exists(ang) or not os.path.exists(ecc) or not os.path.exists(lab):        
            raise ValueError('No areas template found/created for subject: ' + lab)
        angle_mgz = fs.mghformat.load(ang)
        eccen_mgz = fs.mghformat.load(ecc)
        label_mgz = fs.mghformat.load(lab)
        ribbon_mgzs = (subject.mgh_images['lh.ribbon'], subject.mgh_images['rh.ribbon'])
        # The variables are all mgz volumes, so we need to extract the values:
        labels = np.round(np.abs(label_mgz.dataobj.get_unscaled()))
        angles = angle_mgz.dataobj.get_unscaled()
        eccens = eccen_mgz.dataobj.get_unscaled()
        (lrib, rrib) = [r.dataobj.get_unscaled() for r in ribbon_mgzs]
        # Find the voxel indices first:
        # for now we only look at v1-v3
        labels[labels > 3] = 0
        coords = np.asarray(np.where(labels.astype(bool))).T
        # Grab the hemispheres; filter down if something isn't in the ribbon
        tmp   = [(1 if lrib[i,j,k] == 1 else -1, (i,j,k))
                 for (i,j,k) in coords
                 if lrib[i,j,k] != 0 or rrib[i,j,k] != 0
                 if eccens[i,j,k] < max_eccentricity.m]
        hemis  = np.asarray([r[0] for r in tmp], dtype=np.int)
        idcs   = np.asarray([r[1] for r in tmp], dtype=np.int)
        coords = np.asarray(idcs, dtype=np.float)
        # Pull out the angle/eccen data
        angs0   = np.asarray([angles[i,j,k] for (i,j,k) in idcs])
        angles  = angs0*hemis
        eccens  = np.asarray([eccens[i,j,k] for (i,j,k) in idcs], dtype=np.float)
        labels  = np.asarray([labels[i,j,k] for (i,j,k) in idcs], dtype=np.int)
    elif modality.lower() == 'surface':
        rx = freesurfer_subject.RH.midgray_surface.coordinates.T
        lx = freesurfer_subject.LH.midgray_surface.coordinates.T
        # make sure there are template volume files that match this subject
        lang = os.path.join(subject.path, 'surf', 'lh.benson14_angle.mgz')
        lecc = os.path.join(subject.path, 'surf', 'lh.benson14_eccen.mgz')
        llab = os.path.join(subject.path, 'surf', 'lh.benson14_varea.mgz')
        rang = os.path.join(subject.path, 'surf', 'rh.benson14_angle.mgz')
        recc = os.path.join(subject.path, 'surf', 'rh.benson14_eccen.mgz')
        rlab = os.path.join(subject.path, 'surf', 'rh.benson14_varea.mgz')
        (lang,lecc,llab,rang,recc,rlab) = [
            flnm if os.path.isfile(flnm) else flnm[:-4]
            for flnm in (lang,lecc,llab,rang,recc,rlab)]
        if not os.path.exists(lang) or not os.path.exists(rang) or \
           not os.path.exists(lecc) or not os.path.exists(recc) or \
           not os.path.exists(llab) or not os.path.exists(rlab):
            # Apply the template first...
            neurocmd.benson14_retinotopy.main(subject.path)
        if not os.path.exists(lang) or not os.path.exists(rang) or \
           not os.path.exists(lecc) or not os.path.exists(recc) or \
           not os.path.exists(llab) or not os.path.exists(rlab):
            raise ValueError('No anatomical template found/created for subject')
        (lang,lecc,llab,rang,recc,rlab) = [neuro.load(fl) for fl in (lang,lecc,llab,rang,recc,rlab)]
        llab = np.round(np.abs(llab))
        rlab = np.round(np.abs(rlab))
        (angs0, eccs, labs) = [np.concatenate([ldat, rdat], axis=0)
                               for (ldat,rdat) in zip([lang, lecc, llab], [rang, recc, rlab])]
        idcs    = np.concatenate([range(len(lang)), range(len(rang))], axis=0)
        valid   = np.intersect1d(np.intersect1d(np.where(labs > 0)[0], np.where(labs < 4)[0]),
                                 np.where(eccs < max_eccentricity.m)[0])
        idcs    = idcs[valid]
        coords  = np.concatenate([lx, rx], axis=0)[valid]
        hemis   = np.concatenate([[1 for a in lang], [-1 for a in rang]], axis=0)[valid]
        # old versions of the template had positive numbers in both hemispheres...
        if np.mean(angs0[valid[hemis == -1]]) > 0:
            angles  = angs0[valid]*hemis
        else:
            angles = angs0[valid]
        eccens  = eccs[valid]
        labels  = np.asarray(labs[valid], dtype=np.int)
    else:
        raise ValueError('Option modality must be \'surface\' or \'volume\'')
    # do the filtering and convert to pvectors
    if import_filter is None:
        res = {'polar_angles':       units.degree * angles,
               'eccentricities':     units.degree * eccens,
               'labels':             labels,
               'cortex_indices':     idcs,
               'cortex_coordinates': coords,
               'hemispheres':        hemis}
    else:
        sels = [i for (i,(p,e,l,h)) in enumerate(zip(angles, eccens, labels, hemis))
                if import_filter(p,e,l,h)]
        res = {'polar_angles':       units.degree * angles[sels],
               'eccentricities':     units.degree * eccens[sels],
               'labels':             labels[sels],
               'cortex_indices':     idcs[sels],
               'cortex_coordinates': coords[sels],
               'hemispheres':        hemis[sels]}
    # make sure they're all write-only
    for v in res.itervalues():
        v.setflags(write=False)
    return res

@pimms.calc('polar_angles', 'eccentricities', 'labels', 'hemispheres',
            'modality', 'cortex_indices', 'cortex_coordinates', 'cortex_affine',
            cache=True)
def import_retinotopy_data_files(polar_angle_filename, eccentricity_filename, label_filename,
                                 hemisphere_filename=None, import_filter=None,
                                 max_eccentricity=None):
    '''

    import_retinotopy_data_files is a calculation that imports retinotopic map data from a set of
    filenames. The filenames must be MGH (*.mgh, *.mgz) or NifTI (*.nii, *.nii.gz) files, or
    optionally FreeSurfer morph-data files for surface modality, and must
    correspond to the appropriate data file. Each filename may optionally be given as a tuple 
    (lh_filename, rh_filename), in which case the hemisphere is deduced from the file; if a single
    filename is given for each, then either the polar angle data must be negative for the left
    visual field / RH and positive for the right visual field / LH, OR, the optional argument
    hemisphere_filename must be given and must be positive in the left hemisphere and negative in
    the right hemisphere, OR, the label_filename must give positive labels in the RH
    and negative labels in the LH. If both positive and negative values appear in the polar angle 
    data, then those values are always used unmolested, regardless of hemisphere file or label file.

    Note that each volume may contain an array of any dimensions; the voxel addresses are traced as
    the cortex_indices no matter whether the volume is a true 3D volume or, e.g., a 1 x 1 x n
    surface volume. The predictions volume that results always has the same dimensionality as the
    input volume, but the 4th dimension is the number of predictions.
    
    Required afferent parameters:
      @ polar_angle_filename Must give the filename (or an (lh_filename, rh_filename) tuple) of the
        polar angle volume for the model to use; this volume's values must be in units of degrees,
        with 0 degrees of polar angle indicating the upper vertical meridian and +90 degrees of
        polar angle indicating the right horizontal meridian. The polar angle must be positive for
        the right visual field / LH, but may be positive for the left visual field / RH as well if
        negative labels are provided in either the optional hemisphere_filename or in the
        label_filename. The values appearing in the polar angle file always take precedence unless
        no negative values are given here.
      @ eccentricity_filename Must give the filename (or an (lh_filename, rh_filename) tuple) of the
        eccentricity volume for the model to use. This volume's values must be in units of degrees.
      @ label_filename Must give the filename (or an (lh_filename, rh_filename) tuple) of the label
        volume for the model to use. Any value of 0 indicates that that voxel should be ignored in
        all other volume files as well. If both positive and negative values appear in the label
        volume, then the negative values are taken to indicate the right hemisphere and/or the left
        visual fiels.
 
    Optional afferent parameters:
      @ max_eccentricity May specifies the maximum eccentricity value to use (voxels with
        eccentricity values above this are ignored).
      @ hemisphere_filename Map specify the filename of a volume containing positive numbers for
        the LH and negative numbers for the RH. If given, supercededs negative values found in the
        labels file, but not the polar angles file.
      @ import_filter If specified, may give a function that accepts four parameters:
        f(polar_angle, eccentricity, label, hemi); if this function fails to return True for the 
        appropriate values of a particular vertex/voxel, then that vertex/voxel is not included in
        the prediction.

    Provided efferent values:
      @ polar_angles       Polar angle values for each vertex/voxel.
      @ eccentricities     Eccentricity values for each vertex/voxel.
      @ labels             An integer label 1, 2, or 3 for V1, V2, or V3, one per vertex/voxel.
      @ hemispheres        1 if left, -1 if right for all vertex/voxel.
      @ cortex_indices     For vertices, the vertex index (in the appropriate hemisphere) for each;
                           for voxels, the (i,j,k) voxel index for each.
      @ cortex_coordinates For voxels, this is the (i,j,k) voxel index (same as cortex_indices); for
                           surfaces, this is ths (x,y,z) position of each vertex in surface-space.
      @ modality           If the given filenames referred to volume files, then 'volume', otherwise,
                           'surface'.
      @ cortex_affine      The transformation matrix associated with the volume, if any modality
                           is 'volume', otherwise None.

    Notes:
      * polar_angles are always given such that a negative polar angle indicates a RH value and a
        positive polar angle inidicates a LH value
      * labels will always be 1, 2, or 3 indicating V1, V2, or V3

    '''
    # The hemisphere might be specified in a few ways; we look for them by priority.
    # (1) If the filenames are given as (lh, rh):
    flnms = [polar_angle_filename, eccentricity_filename, label_filename]
    if all(isinstance(f, tuple) for f in flnms):
        if hemisphere_filename is not None:
            warnings.warn('ignoring hemisphere_filename input because (lh, rh) filenames given')
        (angs, eccs, lbls) = [(import_mri(l), import_mri(r)) for (l,r) in flnms]
        vshape = angs.shape
        hemis = [np.full(l.shape, v) for (l,v) in zip(lbls, [1, -1])]
        lbls = [np.round(np.abs(l)) for l in lbls]
        (lids, rids) = [np.where((l > 0) & (l < 4)) for l in lbls]
        # extract/concatenate values...
        (angs, eccs, lbls, hemis) = [np.concatenate((l[lids], r[rids]))
                                     for (l,r) in [angs, eccs, lbls, hemis]]
        aids = np.concatenate((np.transpose(lids), np.transpose(rids)))
        crds = None
        tx = None
    elif any(isinstance(f, tuple) for f in flnms):
        raise ValueError('Either all or no filenames must be given as (lh_filename, rh_filename)')
    else:
        (angs, eccs, lbls) = [import_mri(f) for f in flnms]
        vshape = angs.shape
        # get the anatomical id's:
        lbl_sgns = np.sign(lbls)
        lbls = np.round(np.abs(lbls))
        aids = np.where((lbls > 0) & (lbls < 4))
        # extract the data
        (angs, eccs, lbls) = [x[aids] for x in [angs, eccs, lbls]]
        # hemispheres: check first if there are negative values in polar angle data, then check if
        # there are negative labels or if a hemispheres file is given
        if -1 in np.unique(np.sign(angs)):
            hemis = np.sign(angs)
        elif -1 in np.unique(lbl_sgns):
            hemis = lbl_sgns[aids]
        elif hemisphere_filename is not None:
            hemis = import_mri(hemisphere_filename)[aids]
        else:
            # this just means there is only one hemisphere given
            hemis = np.ones(angs.shape, dtype=np.int)
        tx = import_mri(label_filename, 'affine')
        aids = np.asarray(aids, dtype=np.int).T
        crds = np.asarray(aids, dtype=np.float)
    # Okay, we should be set; the modality we can get from the vshape
    modality = 'surface' if len(vshape) == 1 else 'volume'
    return {'polar_angles':  angs,  'eccentricities': eccs,     'labels':             lbls,
            'hemispheres':   hemis, 'cortex_indices': aids,     'cortex_coordinates': crds,
            'cortex_affine': tx,    'modality':       modality}

@pimms.calc('coordinates', memoize=True)
def calc_prediction_coordinates(cortex_coordinates, cortex_affine):
    '''
    calc_prediction_coordinates is a calculation that uses the subject's cortex_affine transform to
    convert between the cortex_coordinates and the coordinates (which represents the coordinates in
    the subject's designated native space). For volume imports, the subject's native space is the
    space designated by the volume's (or freesurfer subject's ribbon's) affine transform. For
    surface imports that come from a freesurfer subject, this is also the space designated by the
    subject's ribbon's affine. If a surface import was used without a freesurfer subject, then
    there is no known space, so coordinates is set to None.

    Provided efferent values:
      @ coordinates Will be the coordinates of the predictions in the orientation specified by the
        import volume's affine transform; if the import was a surface from a freesurfer subject,
        then the space designated by the subject's ribbon's affine is used; if the import was a
        surface not from a freesurfer subject, this will be None.
    '''
    if cortex_coordinates is None or cortex_affine is None: return None
    res = apply_affine(cortex_affine, cortex_coordinates)
    res.setflags(write=False)
    return res

@pimms.calc('predicted_responses_exported_q')
def export_predicted_responses(predicted_responses,
                               predicted_responses_filename, image_ordering_filename,
                               freesurfer_subject, cortex_indices, hemispheres, modality,
                               overwrite_files_on_export=True, export_fill_value=0):
    '''
    export_predicted_responses is a calculation that writes out the results of an SCO model instance
    to disk.

    Required afferent parameters:
      @ predicted_responses Must be a dict of keys (image names) to values (predictions); each value
        must be a vector the same size as cortex_indices.
      @ freesurfer_subject Must be the neuropythy.freesurfer.Subject object.
      @ cortex_indices Must be the indices into the anatomical data; this is generated by the calc
        object import_benson14_from_freesurfer.
      @ hemispheres Must be a vector that is -1 for each RH and 1 for each LH; this is generated
        by the calc object import_benson14_from_freesurfer.
      @ predicted_responses_filename Must be the name of the file to which to write the predicted
        responses; this should be an mgh or mgz file.
      @ image_ordering_filename Must be the name of the file to which to write the stimulus image
        ordering that is used in the predicted_responses_filename; this will be written as text;
        may be None to indicate that no file should be written.
      @ modality Must be either 'surface' or 'volume'.

    Optional afferent parameters:
      @ overwrite_files_on_export May be set to False to prevent files from being overwritten.
      @ export_fill_value May give the value that is filled into the volume/surface file
        on export for voxels or vertices that are not in the V1/V2/V3 ROIs.

    Efferent values:
      @ predicted_responses_exported_q Will be set to True if the responses were successfully
        exported and to False otherwise.
    
    Notes:
      * If modality is 'surface' then 'lh.' and 'rh.' are prepended to the output volume filename;
        each will be a 1 x 1 x n x d volume where n is the number of vertices in the hemisphere and
        d is the number of images
      * The result, predicted_responses_exported_q, is set to False if the files exist and the
        overwrite_files_on_export is False; all other issues raise errors
      * if any file cannot be written and overwrite_files_on_export is False, then no files are
        written
    '''
    # Before we go crazy, let's see if we can write things out at all:
    prfnm = predicted_response_filename
    if not overwrite_files_on_export:
        if image_ordering_filename is not None and os.path.exists(image_ordering_filename):
            return False
        if modality.lower() == 'volume' and os.path.exists(prfnm):
            return False
        elif modality.lower() == 'surface' and \
             not all(os.path.exists(hstr + prfnm) for hstr in ['lh.', 'rh.']):
            return False
    # Okay, now we can start actually doing work:
    fill = float(export_fill_value)
    imorder = predicted_responses.keys()
    if modality.lower() == 'volume':
        vol0 = freesurfer_subject.mgh_images['lh.ribbon']
        vol0dims = vol0.get_data().shape
        preds = np.full(vol0dims + (len(predicted_responses),), fill, dtype=np.float32)
        for (n,imk) in enumerate(imorder):
            for ((i,j,k),val) in zip(cortex_indices, predicted_responses[imk]):
                preds[i,j,k,n] = val
        hdr = vol0.header.copy()
        hdr.set_data_dtype(np.float32)
        hdr.set_data_shape(preds.shape)
        mgh = fs.mghformat.MGHImage(preds, vol0.affine, hdr, vol0.extra, vol0.file_map)
        mgh.to_filename(prfnm)
    else:
        flnms = []
        for (hemi, hid) in zip([subject.LH, subject.RH], [1, -1]):
            idcs  = np.where(hemispheres == hid)[0]
            preds = np.full((1, 1, hemi.vertex_count, len(idcs)), fill, dtype=np.float32)
            pidcs = np.asarray(cortex_indices)[idcs]
            for (i,imk) in enumerate(imorder):
                preds[0,0,pidcs,i] = predicted_responses[imk][idcs]
            mgh = fs.mghformat.MGHImage(preds, np.eye(4))
            mgh_flnm = hemi.name.lower() + '.' + prfnm
            mgh.to_filename(mgh_flnm)
    if image_ordering_filename is not None:
        with open(image_ordering_filename, 'w') as f:
            for im in imorder: f.write('%s\n' % im)
    return True
