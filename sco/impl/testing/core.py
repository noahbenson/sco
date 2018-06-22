####################################################################################################
# sco/impl/testing/core.py
# Implementation of a testing model, which can be used to test most parts of the model easily.
# by Noah C. Benson

import os, sys, six, pimms
import pyrsistent    as pyr
import numpy         as np

import sco.anatomy
import sco.stimulus
import sco.pRF
import sco.contrast
import sco.analysis
import sco.util

from sco.impl.benson17 import (pRF_sigma_slopes_by_label_Kay2013,
                               pRF_sigma_offsets_by_label_Kay2013,
                               contrast_constants_by_label_Kay2013,
                               compressive_constants_by_label_Kay2013,
                               saturation_constants_by_label_Kay2013,
                               divisive_exponents_by_label_Kay2013,
                               spatial_frequency_sensitivity)

@pimms.calc('polar_angles', 'eccentricities', 'labels', 'hemispheres', 'image_retinotopy_indices')
def calc_image_retinotopy(pixels_per_degree, max_eccentricity,
                          output_pixels_per_degree=None,
                          output_max_eccentricity=None):
    '''
    calc_image_retinotopy calculates retinotopic coordinates (polar angle, eccentricity, label) for
    an output image the same size as the input images. I.e., each pixel in the input images get a
    single pRF center for V1, V2, and V3 (one each).
    '''
    maxecc = pimms.mag(max_eccentricity, 'deg') if output_max_eccentricity is None else \
             pimms.mag(output_max_eccentricity, 'deg')
    d2p    = pimms.mag(pixels_per_degree, 'px / deg') if output_pixels_per_degree is None else \
             pimms.mag(output_pixels_per_degree, 'px / deg')
    dim    = int(np.round(d2p * 2.0 * maxecc))
    center = 0.5 * (dim - 1) # in pixels
    # x/y in pixels
    xi = np.asarray(range(dim))
    # x/y in degrees
    x = (xi - center) / d2p
    # mesh grid...
    (x,y) = np.meshgrid(x,-x)
    # convert to eccen and theta
    eccen = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    # and to angle
    angle = np.mod(90 - 180.0/np.pi*theta + 180, 360.0) - 180
    # get hemispheres
    hemis = np.sign(angle)
    hemis[hemis == 0] = 1.0
    # and turn these into lists with visual area labels
    (angle, eccen, hemis) = [u.flatten()             for u in (angle, eccen, hemis)]
    label = np.concatenate([np.full(len(angle), k) for k in (1,2,3)])
    (angle, eccen, hemis) = [np.concatenate((u,u,u)) for u in (angle, eccen, hemis)]
    # make the indices
    (ci,ri) = np.meshgrid(xi,xi)
    idcs = np.transpose((ri.flatten(), ci.flatten()))
    for u in (angle, eccen, label, hemis, idcs):
        u.setflags(write=False)
    return {'polar_angles':             pimms.quant(angle, 'deg'),
            'eccentricities':           pimms.quant(eccen, 'deg'),
            'labels':                   label,
            'hemispheres':              hemis,
            'image_retinotopy_indices': idcs}

def create_test_stimuli(stimulus_directory, max_eccentricity=10.0, pixels_per_degree=6.4,
                        orientations=None, spatial_frequencies=None, contrasts=None,
                        modulated_orientations=None, modulated_spatial_frequency=1.0):
    '''
    create_test_stimuli(stimulus_directory) creates a variety of test png files in the given
      stimulus_directory. The image filenames are returned as a list. The images created are as
      follows:
      * A blank gray image
      * A variety of sinusoidal gratings; these vary in terms of:
        * contrast
        * spatial frequency
        * orientation
      * A variety of modulated sinusoidal gratings; in which the modulated spatial frequency is
        held at 1 cyce / degree and the modulated contrast is as high as possible; these vary in
        terms of:
        * contrast
        * spatial frequency
        * orientation
        * modulated orientation

    The following options may be given:
      * max_eccentricity (default: 10.0) specifies the maximum eccentricity in degrees to include in
        the generated images.
      * pixels_per_degree (default: 6.4) specifies the pixels per degree of the created images.
      * orientations (default: None) specifies the orientations (NOTE: in degrees) of the various
        gratings generated; if None, then uses [0, 30, 60, 90, 120, 150].
      * spatial_frequencies (defaut: None) specifies the spatial frequencies to use; by default uses
        a set of 5 spatial frequencies that depend on the resolution specified by pixels_per_degree.
      * contrasts (default: None) specifies the contrasts of the images to make; if None, then uses
        [0.333, 0.667, 1.0].
    '''
    import skimage.io, warnings
    # Process/organize arguments
    maxecc = pimms.mag(max_eccentricity,  'deg')
    d2p    = pimms.mag(pixels_per_degree, 'px/deg')
    sdir   = stimulus_directory
    thetas = np.arange(0, 180, 30) if orientations is None else np.asarray(orientations)
    modths = np.arange(0, 180, 45) if modulated_orientations is None else \
             np.asarray(moduated_orientations)
    sfreqs = (d2p/2)*(2**np.linspace(-4.0, 0.0, 5)) if spatial_frequencies is None else \
             spatial_frequencies
    modsf  = pimms.mag(modulated_spatial_frequency, 'cycles/degree')
    ctsts  = [0.333, 0.667, 1.0] if contrasts is None else contrasts
    # go ahead and setup x and y values (in degrees) for the images
    dim    = np.round(d2p * 2.0 * maxecc)
    center = 0.5 * (dim - 1) # in pixels
    # x/y in pixels
    x = np.arange(0, dim, 1)
    # x/y in degrees
    x = (x - center) / d2p
    # mesh grid...
    (x,y) = np.meshgrid(x,x)
    # how we export
    fldat = []
    flnm0 = 'grating=%s_th=%s_mt=%s_sf=%s_ct=%5.3f.png'
    def _imsave(im, gr, th, mt, sf, ct):
        flnm = ('blank.png' if gr is None else
                (flnm0 % (gr,
                          'excluded' if th is None else ('%05.1fdeg' % th),
                          'excluded' if mt is None else ('%05.1fdeg' % mt),
                          'excluded' if sf is None else ('%05.3fdeg' % sf),
                          ct)))
        flnm = os.path.join(sdir, flnm)
        skimage.io.imsave(flnm, np.asarray(np.round(np.clip(im, 0, 1) * 65535), dtype=np.uint16))
        fldat.append({'grating':           gr,
                      'contrast':          (np.nan if th is None else ct),
                      'theta':             (np.nan if th is None else th),
                      'modulated_theta':   (np.nan if mt is None else mt),
                      'spatial_frequency': (np.nan if sf is None else sf),
                      'filename':          flnm})
    # have to catch the UserWarnings for low contrast images
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=UserWarning)
        # Start with a simple blank image
        _imsave(np.full((128, 128), 0.5), None, None, None, None, None)
        # Generate the basic contrast images
        for theta_deg in thetas:
            theta = np.pi/180 * theta_deg
            for cpd in sfreqs:
                cpp = cpd / d2p
                # Make the sine-grating first
                im0 = 0.5 * (1 + np.sin((np.cos(theta)*x - np.sin(theta)*y) * 2 * np.pi * cpp))
                # now, write out a simple square-grating version
                _imsave(np.round(im0), 'sqr', theta_deg, None, cpd, 1.0)
                # Now we can look at the requested contrasts
                for ct in ctsts:
                    # save the simple grating first
                    im = (im0 - 0.5)*ct + 0.5
                    _imsave(im, 'sin', theta_deg, None, cpd, ct)
                    # then we want to look at the modulated gratings
                    for modth_deg in modths:
                        # need a new grating...
                        const = 2 * np.pi * modsf / d2p
                        modth = np.pi/180.0 * modth_deg
                        modim = 0.5 * (1 + np.sin((np.cos(modth)*x - np.sin(modth)*y) * const))
                        _imsave((im - 0.5)*modim + 0.5, 'sin', theta_deg, modth_deg, cpd, ct)
        # Make fldat into a pimms itable
        tbl = pimms.itable({k:np.asarray([ff[k] for ff in fldat]) for k in six.iterkeys(fldat[0])})
        return tbl

@pimms.calc('stimulus', 'stimulus_metadata', 'temporary_directory')
def calc_stimulus(max_eccentricity, pixels_per_degree,
                  stimulus_directory=None):
    '''
    calc_stimulus calculates a set of stimulus files; these are saved to the folder given by the
    afferent parameter stimulus_directory, which may be left as None (it's default value) in order
    to use a temporary directory. The calculator will first attempt to create a variety of files
    with different basic parameters in this directory then will load all png files in the directory.
    '''
    import tempfile, skimage.io
    if stimulus_directory is None:
        tempdir = tempfile.mkdtemp(prefix='sco_testing_')
        stimulus_directory = tempdir
    else:
        tempdir = None
    # Okay, make the files
    fldat = create_test_stimuli(stimulus_directory,
                                max_eccentricity=max_eccentricity,
                                pixels_per_degree=pixels_per_degree)
    return (tuple(fldat['filename']), fldat, tempdir)

# Default Options ##################################################################################
# The default options are provided here for the SCO
@pimms.calc('testing_default_options_used')
def provide_default_options(
        pRF_sigma_slopes_by_label              = pRF_sigma_slopes_by_label_Kay2013,
        pRF_sigma_offsets_by_label             = pRF_sigma_offsets_by_label_Kay2013,
        contrast_constants_by_label            = contrast_constants_by_label_Kay2013,
        compressive_constants_by_label         = compressive_constants_by_label_Kay2013,
        saturation_constants_by_label          = saturation_constants_by_label_Kay2013,
        divisive_exponents_by_label            = divisive_exponents_by_label_Kay2013,
        max_eccentricity                       = pimms.quant(10.0, 'deg'),
        pixels_per_degree                      = pimms.quant(6.4, 'px/deg'),
        modality                               = 'surface',
        spatial_frequency_sensitivity_function = spatial_frequency_sensitivity):
    '''
    provide_default_options is a calculator that optionally accepts values for all parameters for
    which default values are provided in the sco.impl.testing package and yields into the calc plan
    these parameter values or the default ones for any not provided.
 
    These options are:
      * pRF_sigma_slope_by_label (sco.impl.benson17.pRF_sigma_slope_by_label_Kay2013)
      * compressive_constant_by_label (sco.impl.benson17.compressive_constant_by_label_Kay2013)
      * contrast_constant_by_label (sco.impl.benson17.contrast_constant_by_label_Kay2013)
      * modality ('surface')
      * max_eccentricity (10)
      * spatial_frequency_sensitivity_function (from sco.impl.benson17)
      * saturation_constant (sco.impl.benson17.saturation_constant_Kay2013)
      * divisive_exponent (sco.impl.benson17.divisive_exponent_Kay2013)
      * gabor_orientations (8)
    '''
    # the defaults are filled-in by virtue of being in the above argument list
    return True

# The volume (default) calculation chain
sco_plan_data = pyr.pmap(
    {k:v
     for pd    in [sco.stimulus.stimulus_plan_data,
                   sco.contrast.contrast_plan_data,
                   sco.pRF.pRF_plan_data,
                   #sco.anatomy.anatomy_plan_data,
                   #sco.analysis.analysis_plan_data,
                   #sco.util.export_plan_data,
                   {'default_options':  provide_default_options,
                    'stimulus':         calc_stimulus,
                    'image_retinotopy': calc_image_retinotopy}]
     for (k,v) in pd.iteritems()})

sco_plan      = pimms.plan(sco_plan_data)
