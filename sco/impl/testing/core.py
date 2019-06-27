####################################################################################################
# sco/impl/testing/core.py
# Implementation of a testing model, which can be used to test most parts of the model easily.
# by Noah C. Benson

import os, sys, six, pimms
import pyrsistent    as pyr
import numpy         as np
import scipy         as sp

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
                               gains_by_label_Kay2013,
                               spatial_frequency_sensitivity)

def imsave(flnm, image):
    '''
    imsave is a wapper around skimage.io.imsave, but convertes floating-point images into integers.
    '''
    import skimage.io
    image = np.asarray(np.round(np.clip(image, 0, 1) * 65535), dtype=np.uint16)
    return skimage.io.imsave(flnm, image)
def luminance_grating(pixels=128, cpp=0.46875, theta=0, contrast=1):
    '''
    Yields an image with a luminance grating.
    '''
    x = np.arange(0, pixels, 1)
    center = 0.5 * (x - 1)
    # x/y:
    x -= center
    # mesh grid...
    (x,y) = np.meshgrid(x,x)
    im = contrast * 0.5 * (1 + np.sin((np.cos(theta)*x - np.sin(theta)*y) * 4 * np.pi * cpp))
    return im
def test_grating(max_eccentricity=10, pixels_per_degree=6.4, theta=0, mod_theta=np.pi/2,
                 spatial_frequency=3, mod_spatial_frequency=None, contrast=1, mod_contrast=0):
    '''
    test_grating() yields a test-image; this is typically an image with a single luminance
      grating that is modulated by a contrast grating. By default the contrast grating is flat, so
      there is effectively no contrast grating, but this any many other features of the image may
      be manipulated via the various optional parameters.

    Optional arguments:
      * max_eccentricity (default: 10 degrees) specifies the max eccentricity to include in the
        image; this is used in combination with pixels_per_degree.
      * pixels_per_dgree (default: 6.4 px/deg) specifies the number of pixels per degree.
      * theta (default: 0) specifies the angle of the luminance gradient.
      * mod_theta (default: pi/2) specifies the angle of the modulating contrast gradient.
      * spatial_frequency (default: 3 cycles/deg) specifies the spatial frequency of the luminance
        grating.
      * mod_spatial_frequency (default: 0.5 cycles/deg) specifies the spatial frequency of the
        modulating contrast grating.
      * contrast (default: 1) specifies the contrast of the luminance grating.
      * mod_contrast (default: 0) specifies the amount of contrast in the modulated grating. Note
        that the default value of 0 will result in an image without a modulated contrast grating.
    '''
    import warnings
    # Process/organize arguments
    maxecc = pimms.mag(max_eccentricity,  'deg')
    d2p    = pimms.mag(pixels_per_degree, 'px/deg')
    theta  = pimms.mag(theta, 'rad')
    modth  = pimms.mag(mod_theta, 'rad')
    cpp    = pimms.mag(spatial_frequency, 'turns/deg') / d2p
    # go ahead and setup x and y values (in degrees) for the images
    dim    = np.round(d2p * 2.0 * maxecc)
    # Luminance grating
    im = luminance_grating(pixels=dim, cpp=cpp, theta=theta, contrast=contrast)
    # Contrast grating
    if None not in [modth, mod_spatial_frequency, mod_contrast]:
        mod_cpp = pimms.mag(mod_spatial_frequency, 'turns/deg') / d2p
        mod_im = luminance_grating(dim, cpp=mod_cpp, theta=mod_theta, contrast=mod_contrast)
        # multiply the two together
        im = (im - 0.5)*mod_im + 0.5
    return im
def make_disk(imsz, radius, mid=None, edge_width=0):
    '''
    make_disk(imsize, radius, centerpx, edge_width) yields a disk matrix whose internal values
      are 1 and whose external values are 0; the image size, radius of the disk, and the disk's
      center are given in pixels as the first three arguments. The final argument, edge_width
      is the number of pixels over which to smoothly blend the edge of the disk.
      
    If the parameter centerpx is omitted, the center of the image is used automatically.
    '''
    if pimms.is_number(imsz): imsz = (int(np.round(imsz)), int(np.round(imsz)))
    if   mid is None:          mid = ((imsz[0] - 1)/2, (imsz[1] - 1)/2)
    elif pimms.is_number(mid): mid = (mid, mid)
    if edge_width is None: edge_width = 0
    (x,y) = [np.arange(-u, sz - u, 1) for (u,sz) in zip(mid,imsz)]
    r = np.sqrt(np.sum([u**2 for u in np.meshgrid(x, y)], axis=0))
    im = (r < radius).astype('float')
    if not np.isclose(edge_width, 0) and edge_width > 0:
        (r0,r1) = (radius - edge_width/2, radius + edge_width/2)
        ii = (r > r0) & (r < r1)
        im[ii] = 0.5*(1 - np.sin((r[ii] - radius) / edge_width * np.pi))
    return im
def gaussian_2d(sz, sig):
    '''
    gaussian_2d(size, sigma) yields a matrix with the given size representing a 2D gaussian
      image with the given sigma as its standard deviation.
    '''
    sz = int(np.round(sz))
    mid = (sz - 1)/2
    if not pimms.is_number(mid): mid = np.mean(mid)
    x = np.arange(-mid, sz - mid, 1)
    r2 = np.sum([u**2 for u in np.meshgrid(x, x)], axis=0)
    res = np.exp(-0.5 * r2 / sig**2)
    return res / np.sum(res)
def noise_pattern_stimulus(imsz, density=20, contrast=0.5, sigma0=3):
    '''
    noise_pattern_stimulus(size) generates a noise-pattern image of the given size.
    noise_pattern_stimulus(size, density) uses the given density of curve (default: 20).
    noise_pattern_stimulus(size, density, contrast) uses the given contrast.

    The noise_pattern_stimulus function is largely lifted from the BAIR_stimuli github repository:
    https://github.com/BAIRR01/BAIR_stimuli/ -- specifically the file createPatternStimulus.m.
    '''
    # Create a random seed
    im = np.random.normal(size=(imsz,imsz));
    im = im / (np.max(im - np.min(im))) + 0.5;
    # Create a soft round filter in Fourier space
    radius = density;
    mask = make_disk(im.shape, radius, None, radius/5.0);
    # Filter the image
    dft = np.fft.fftshift(np.fft.fft2(im));
    dft = mask * dft;
    res = np.fft.ifft2(np.fft.ifftshift(dft));
    # Threshold the filtered noise
    thresh = (res - np.min(res)) > (np.max(res) - np.min(res))/2;
    thresh = thresh.astype('float')
    # Grab edges with derivative filter
    edge1 = [[0, 0, 0], [1, 0, -1], [0, 0, 0]];
    edge2 = [[0, 1, 0], [0, 0, 0], [0, -1, 0]];
    edge  = -(sp.ndimage.convolve(thresh, edge1, mode='wrap')**2 + 
              sp.ndimage.convolve(thresh, edge2, mode='wrap')**2)
    # Scale the edge contrast
    edge = contrast*edge; 
    # Filter convolutionally with bpfilter in the image domain
    # the 0.011 and 1.5 were determined by trial and error; this code was lifted from
    # the BAIR_Stimuli repository (https://github.com/BAIRR01/BAIR_stimuli), file
    # stimMakeBandPassFilter.m
    csig = 0.011 / sigma0 * np.min(imsz)
    ssig = sigma0 * 1.5;
    gsz  = round(10 * ssig);
    cfilt = gaussian_2d(gsz, csig)
    sfilt = gaussian_2d(gsz, ssig)
    bp_filt = cfilt / np.mean(cfilt) - sfilt / np.mean(sfilt)
    bp_filt /= np.sum(np.abs(bp_filt))
    output = sp.ndimage.convolve(edge, bp_filt, mode='wrap')
    return np.clip(output + 0.5, 0, 1)
def test_stimuli(stimulus_directory, max_eccentricity=10.0, pixels_per_degree=6.4,
                 orientations=None, spatial_frequencies=None, contrasts=None, densities=None,
                 base_density=10, base_contrast=0.75,
                 base_spatial_frequency=0.8, base_orientation=0):
    '''
    test_stimuli(stimulus_directory) creates a variety of test png files in the given directory.
      The image filenames are returned as a list. The images created are as follows:
      * A blank gray image
      * A variety of sinusoidal gratings; these vary in terms of:
        * contrast
        * spatial frequency
        * orientation
      * A variety of modulated sinusoidal gratings; in which the modulated spatial frequency is
        held at 1 cycle / degree and the modulated contrast is as high as possible; these vary in
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
        [0.25, 0.5, 0.75, 1.0].
    '''
    import skimage.io, warnings
    # Process/organize arguments
    maxecc = pimms.mag(max_eccentricity,  'deg')
    d2p    = pimms.mag(pixels_per_degree, 'px/deg')
    sdir   = stimulus_directory
    thetas = np.arange(0, 180, 22.5) if orientations is None else np.asarray(orientations)
    sfreqs = (d2p/4) * (2**np.linspace(-4.0, 0.0, 5)) if spatial_frequencies is None else \
             pimms.mag(spatial_frequencies, 'cycles/degree')
    ctsts  = [0.25, 0.5, 0.75, 1.0] if contrasts is None else contrasts
    dnsts  = [3, 6, 12, 24, 48] if densities is None else densities
    cpd0   = pimms.mag(base_spatial_frequency, 'cycles/degree')
    th0    = pimms.mag(base_orientation, 'deg') * np.pi/180
    ct0    = base_contrast
    dn0    = base_density
    # go ahead and setup x and y values (in degrees) for the images
    dim    = np.round(d2p * 2.0 * maxecc)
    center = 0.5 * (dim - 1) # in pixels
    # x/y in pixels
    x = np.arange(0, dim, 1)
    # x/y in degrees
    x = (x - center) / d2p
    # mesh grid...
    (x,y) = np.meshgrid(x,x)
    # how we save images
    flnms = []
    fldat = []
    def _imsave(nm, im):
        flnm = os.path.join(sdir, nm + '.png')
        flnms.append(flnm)
        return imsave(flnm, im)
    def _immeta(meta, **kw):
        meta = pimms.merge(meta, kw)
        fldat.append(meta)
        return meta
    # have to catch the UserWarnings for low contrast images
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=UserWarning)
        # Generate the basic sin-grating-based images
        # First we want varying orientation, all at cpd=1
        meta = {'cpd':cpd0, 'contrast':ct0, 'type':'sin_orientation', 'theta':th0}
        for theta_deg in sorted(thetas):
            theta = np.pi/180 * theta_deg
            im = 0.5 * (1 + np.sin((np.cos(theta)*x - np.sin(theta)*y) * 2 * np.pi * cpd0))
            if ct0 < 1: im = (im - 0.5) * ct0  + 0.5
            _imsave('sin[theta=%06.3f]' % theta, im)
            _immeta(meta, theta=theta)
        # okay, now for theta=0 and variable cpd
        for cpd in reversed(sorted(sfreqs)):
            im = 0.5 * (1 + np.sin((np.cos(th0)*x - np.sin(th0)*y) * 2 * np.pi * cpd))
            if ct0 < 1: im = (im - 0.5) * ct0  + 0.5
            # write it out
            _imsave('sin[cpd=%06.3f]' % cpd, im)
            _immeta(meta, cpd=cpd, type='sin_frequency')
        # Now we can look at the requested contrasts
        for ct in sorted(ctsts):
            # save the simple grating first
            im = 0.5 * (1 + np.sin((np.cos(th0)*x - np.sin(th0)*y) * 2 * np.pi * cpd0))
            if ct < 1: im = (im - 0.5) * ct + 0.5
            _imsave('sin[contrast=%06.3f]' % ct, im)
            _immeta(meta, contrast=ct, type='sin_contrast')
        # Next, plaids; these are relatively easy:
        x0 = np.cos(th0)*x - np.sin(th0)*y
        y0 = np.sin(th0)*x + np.cos(th0)*y
        im0 = 0.25 * (2 + np.sin(x0 * 2 * np.pi * cpd0) + np.sin(y0 * 2 * np.pi * cpd0))
        im0 = (im0 - 0.5) / np.max(np.abs(im0 - 0.5)) + 0.5
        for ct in sorted(ctsts):
            if ct == 0: continue
            if ct < 1: im = (im0 - 0.5) * ct + 0.5
            else:      im = im0
            _imsave('plaid[contrast=%06.3f]' % ct, im)
            _immeta(meta, contrast=ct, type='plaid_contrast')
        # okay, now the sum of sixteen gratings
        const = 2*np.pi*cpd0
        for ct in sorted(ctsts):
            if ct == 0: continue
            im0 = np.mean(
                [0.5*(1 + np.sin(const*(np.random.rand()-0.5 + np.cos(th)*x - np.sin(th)*y)))
                 for th in np.linspace(0, 2*np.pi, 17)[:-1]],
                axis=0)
            im0 = (im0 - 0.5) / np.max(np.abs(im0 - 0.5)) + 0.5
            if ct < 1: im = (im0 - 0.5) * ct + 0.5
            else:      im = im0
            _imsave('circ[contrast=%06.3f]' % ct, im)
            _immeta(meta, contrast=ct, type='circ_contrast')
        # Okay, make the noise pattern images
        dmeta = pyr.pmap(meta).discard('orientation').set('cpd', cpd0)
        for dn in dnsts:
            if dn <= 0: continue
            im = noise_pattern_stimulus(x.shape[0], dn, ct0, cpd0/2)
            _imsave('noise[density=%02d]' % dn, im)
            _immeta(dmeta, density=dn, type='noise_density')
    # Make fldat into a pimms itable
    fldat = [m.set('filename', flnm) for (m,flnm) in zip(fldat, flnms)]
    tbl = pimms.itable({k:np.asarray([ff[k] for ff in fldat]) for k in six.iterkeys(fldat[0])})
    return tbl

@pimms.calc('polar_angles', 'eccentricities', 'labels', 'hemispheres')
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
    # we progressively downsample in order to make this more efficient...
    (ang,ecc,hem) = ([],[],[])
    esteps = np.concatenate([[0], 1.5 * 2**np.arange(0, np.ceil(np.log2(maxecc)), 1)])
    for (k0,k1) in zip(esteps[:-1], esteps[1:]):
        # make the image-data at resolution k1; we can ignore beyond k1 eccen for now
        dim    = int(np.ceil((d2p / k1) * 2.0 * k1))
        center = 0.5 * (dim - 1) # in pixels
        # x/y in degrees
        x = (np.arange(dim) - center) / (d2p/k1)
        # mesh grid...
        (x,y) = np.meshgrid(x,-x)
        # convert to eccen and theta/angle
        eccen = np.sqrt(x**2 + y**2)
        theta = np.arctan2(y, x)
        angle = np.mod(90 - 180.0/np.pi*theta + 180, 360.0) - 180
        # get hemispheres
        hemis = np.sign(angle)
        hemis[hemis == 0] = 1.0
        # find the relevant pixels
        ii = (k0 <= eccen) & (eccen < k1)
        ang.append(angle[ii])
        ecc.append(eccen[ii])
        hem.append(hemis[ii])
    # and turn these into lists with visual area labels
    (angle, eccen, hemis) = [np.concatenate(u) for u in (ang, ecc, hem)]
    label = np.concatenate([np.full(len(angle), k) for k in (1,2,3)])
    (angle, eccen, hemis) = [np.concatenate((u,u,u)) for u in (angle, eccen, hemis)]
    for u in (angle, eccen, label, hemis): u.setflags(write=False)
    return {'polar_angles':             pimms.quant(angle, 'deg'),
            'eccentricities':           pimms.quant(eccen, 'deg'),
            'labels':                   label,
            'hemispheres':              hemis}

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
    fldat = test_stimuli(stimulus_directory,
                         max_eccentricity=max_eccentricity,
                         pixels_per_degree=pixels_per_degree)
    return (tuple(fldat['filename']), fldat, tempdir)

# Default Options ##################################################################################
# The default options are provided here for the SCO
@pimms.calc('testing_default_options_used')
def provide_default_options(
        pRF_sigma_slopes_by_label      = pRF_sigma_slopes_by_label_Kay2013,
        pRF_sigma_offsets_by_label     = pRF_sigma_offsets_by_label_Kay2013,
        contrast_constants_by_label    = contrast_constants_by_label_Kay2013,
        compressive_constants_by_label = compressive_constants_by_label_Kay2013,
        saturation_constants_by_label  = saturation_constants_by_label_Kay2013,
        divisive_exponents_by_label    = divisive_exponents_by_label_Kay2013,
        gains_by_label                 = gains_by_label_Kay2013,
        max_eccentricity               = pimms.quant(10.0, 'deg'),
        pixels_per_degree              = pimms.quant(6.4, 'px/deg'),
        modality                       = 'surface',
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
     for (k,v) in six.iteritems(pd)})

sco_plan      = pimms.plan(sco_plan_data)
