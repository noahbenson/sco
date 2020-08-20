####################################################################################################
# contrast/core.py
# The second-order contrast module of the standard cortical observer.
# By Noah C. Benson

import numpy                 as     np
import numpy.matlib          as     npml
import pyrsistent            as     pyr
import neuropythy            as     ny
import skimage.transform     as     sktr
from   scipy                 import ndimage as ndi
from   ..util                import (lookup_labels, units, global_lookup, gabor_kernel)
from   .spyr                 import spyr_filter

import pimms

@pimms.calc('gabor_spatial_frequencies')
def calc_gabor_spatial_frequencies(image_array, pixels_per_degree,
                                   gabor_spatial_frequency_count=16,
                                   gabor_spatial_frequency_min=None,
                                   gabor_spatial_frequency_max=None):
    '''
    calc_gabor_spatial_frequencies is a calculator that constructs a reasonable set of spatial
    frequencies (in cycles per degree) at which to filter an image given its pixels per degree. The
    reason for this calculator is that for a low-contrast image, a given frequency, even if visible
    to a human observer in theory, might be so high that we can't detect it at the given image
    resolution. This function picks a set of <gabor_spatial_frequency_count> frequencies that are
    evenly log-spaced such that no frequency is smaller than a minimum or larger than a maximum that
    are chosen based on image resolution.

    Afferent parameters:
      * image_array
      * pixels_per_degree
      @ gabor_spatial_frequency_count Must be the number of spatial frequencies at which to filter
        the images; by default this is 16.
      @ gabor_spatial_frequency_min Must be the minimum spatial frequency (in cycles per degree) at
        which to filter the images. If None, then the minimum is equivalent to one half of a cycle
        per image. By default, this is Ellipsis.
      @ gabor_spatial_frequency_max Must be the maximum spatial frequency (in cycles per degree) at
        which to filter the images. If None, then the maximum is equivalent to 0.5 cycles / pixel.

    Efferent values:
      @ gabor_spatial_frequencies Will be a numpy array of spatial frequency values to use in the
        filtering of the image arrays; values will be in cycles/degree.
    '''
    # Process some arguments
    pixels_per_degree = pimms.mag(pixels_per_degree, 'pixels/degree')
    imsz = image_array.shape[1]
    imsz_deg = imsz / pixels_per_degree
    if gabor_spatial_frequency_min is None:
        gabor_spatial_frequency_min = 0.5 / imsz_deg
    gabor_spatial_frequency_min = pimms.mag(gabor_spatial_frequency_min, 'cycles/degree')
    if gabor_spatial_frequency_min == 0:
        raise ValueError('gabor_spatial_frequency_min must be positive')
    if gabor_spatial_frequency_max is None:
        gabor_spatial_frequency_max = 0.5 * pixels_per_degree
    gabor_spatial_frequency_max = pimms.mag(gabor_spatial_frequency_max, 'cycles/degree')
    if np.isinf(gabor_spatial_frequency_max):
        raise ValueError('gabor_spatial_frequency_min must be finite')
    # okay, lets transfer to log-space
    lmin = np.log(gabor_spatial_frequency_min)
    lmax = np.log(gabor_spatial_frequency_max)
    # make the spacing
    lfs = np.linspace(lmin, lmax, gabor_spatial_frequency_count)
    # return to exponential space
    fs = np.exp(lfs)
    # That's basically it!
    fs.setflags(write=False)
    return pimms.quant(fs, 'cycles/degree')

def _convolve_from_arg(arg):
    'Private function used for multiprocessing the convolution of images.'
    (imgs, kern, bg) = arg
    re = np.asarray([ndi.convolve(img, kern.real, mode='constant', cval=bg) for img in imgs])
    im = np.asarray([ndi.convolve(img, kern.imag, mode='constant', cval=bg) for img in imgs])
    return re + 1j*im
def _spyr_from_arg(arg):
    'Private function used for multiprocessing the steerable-pyramid filtering of images.'
    (imgs, th, cpp, nth) = arg
    return spyr_filter(imgs, th, cpp, 1, nth)
_default_gabor_orientations        = pimms.quant(np.asarray(range(8), dtype=np.float) * np.pi/8.0,
                                                 'rad')

@pimms.calc('oriented_contrast_images', 'scaled_pixels_per_degree', 'scaled_image_arrays',
            'gabor_filters',
            cache=True)
def calc_oriented_contrast_images(image_array, pixels_per_degree, background,
                                  gabor_spatial_frequencies,
                                  gabor_orientations=_default_gabor_orientations,
                                  max_image_size=250,
                                  min_pixels_per_degree=0.5, max_pixels_per_filter=27,
                                  ideal_filter_size=17,
                                  use_spatial_gabors=True,
                                  multiprocess=True):
    '''
    calc_oriented_contrast_images is a calculator that takes as input an image array along with its
    resolution (pixels_per_degree) and a set of gabor orientations and spatial frequencies, and
    computes the result of filtering the images in the image array with the various filters. The
    results of this calculation are stored in oriented_contrast_images, which is an ordered tuple of
    contrast image arrays; the elements of the tuple correspond to spatial frequencies, which are
    given by the afferent value gabor_spatial_frequencies, and the image arrays are o x n x n where
    o is the number of orientations, given by the afferent value gabor_orientations, and n is the
    size of the height/width of the image at that particular spatial frequency. Note that the
    scaled_pixels_per_degree gives the pixels per degree of all of the elements of the efferent
    value oriented_contrast_energy_images.

    Required afferent values:
      * image_array
      * pixels_per_degree
      * background
      @ use_spatial_gabors Must be either True (use spatial gabor filters instead of the steerable
        pyramid) or False (use the steerable pyramid); by default this is True.
      @ gabor_orientations Must be a list of orientation angles for the Gabor filters or an integer
        number of evenly-spaced gabor filters to use. By default this is equivalent to 8.
      @ gabor_spatial_frequencies Must be a list of spatial frequencies (in cycles per degree) to
        use when filtering the images.
      @ min_pixels_per_degree Must be the minimum number of pixels per degree that will be used to
        represent downsampled images during contrast energy filtering. If this number is too low,
        the resulting filters may lose precision, while if this number is too high, the filtering
        may require a long time and a lot of memory. The default value is 0.5.
      @ max_pixels_per_filter Must be the maximum size of a filter before down-sampling the image
        during contrast energy filtering. If this value is None, then no rescaling is done during
        filtering (this may require large amounts of time and memory). By default this is 27. Note
        that min_pixels_per_degree has a higher precedence than this value in determining if an
        image is to be resized.
      @ ideal_filter_size May specify the ideal size of a filter when rescaling. By default this is
        17.
      @ max_image_size Specifies that if the image array has a dimension with more pixels thatn the
        given value, the images should be down-sampled.

    Efferent output values:
      @ oriented_contrast_images Will be a tuple of n image arrays, each of which is of size
        q x m x m, where n is the number of spatial frequencies, q is the number of orientations,
        and m is the size of the scaled image in pixels. The scaled image pixels-per-degree values
        are given by scaled_pixels_per_degree. The values in the arrays represent the complex-valued
        results of filtering the image_array with the specified filters.
      @ scaled_image_arrays Will be a tuple of scaled image arrays; these are scaled to different
        sizes for efficiency in filtering and are stored here for debugging purposes.
      @ scaled_pixels_per_degree Will be a tuple of values in pixels per degree that specify the
        resolutions of the oriented_contrast_energy_images.
      @ gabor_filters Will be a tuple of gabor filters used on each of the scaled_image_arrays.

    '''
    # process some arguments
    if pimms.is_number(gabor_orientations):
        gabor_orientations = np.pi * np.arange(0, gabor_orientations) / gabor_orientations
    gabor_orientations = np.asarray([pimms.mag(th, 'rad') for th in gabor_orientations])
    nth = len(gabor_orientations)
    if min_pixels_per_degree is None: min_pixels_per_degree = 0
    min_pixels_per_degree = pimms.mag(min_pixels_per_degree, 'pixels / degree')
    if max_pixels_per_filter is None: max_pixels_per_filter = image_array.shape[-1]
    max_pixels_per_filter = pimms.mag(max_pixels_per_filter, 'pixels')
    # if the ideal filter size is given as None, make one up
    if ideal_filter_size is None: ideal_filter_size = (max_pixels_per_filter + 1) / 2 + 1
    ideal_filter_size = pimms.mag(ideal_filter_size, 'pixels')
    if ideal_filter_size > max_pixels_per_filter: ideal_filter_size = max_pixels_per_filter
    pool = None
    if multiprocess is True or pimms.is_int(multiprocess):
        try:
            import multiprocessing as mp
            pool = mp.Pool(mp.cpu_count() if multiprocess is True else multiprocess)
        except: pool = None
    # These will be updated as we go through the spatial frequencies:
    d2p0 = pimms.mag(pixels_per_degree, 'pixels/degree')
    imar = image_array
    d2p  = d2p0
    # how wide are the images in degrees
    imsz_deg = float(image_array.shape[-1]) / d2p0
    # If we're over the max image size, we need to downsample now
    if image_array.shape[1] > max_image_size:
        ideal_zoom = image_array.shape[-1] / max_image_size
        imar = sktr.pyramid_reduce(imar.T, ideal_zoom,
                                   mode='constant', cval=background, multichannel=True).T
        d2p = float(imar.shape[1]) / imsz_deg
    # We build up these  solution
    oces = {} # oriented contrast energy images
    d2ps = {} # scaled pixels per degree
    sims = {} # scaled image arrays
    flts = {} # gabor filters
    # walk through spatial frequencies first
    for cpd in reversed(sorted(gabor_spatial_frequencies)):
        cpd = pimms.mag(cpd, 'cycles/degree')
        cpp = cpd / d2p
        # start by making the filter...
        filt = gabor_kernel(cpp, theta=0)
        # okay, if the filt is smaller than max_pixels_per_filter, we're fine
        if filt.shape[0] > max_pixels_per_filter and d2p > min_pixels_per_degree:
            # we need to potentially resize the image, but only if it's still
            # higher resolution than min pixels per degree
            ideal_zoom = float(filt.shape[0]) / float(ideal_filter_size)
            # resize an image and check it out...
            im = sktr.pyramid_reduce(imar[0], ideal_zoom, mode='constant', cval=background,
                                     multichannel=False)
            new_d2p = float(im.shape[0]) / imsz_deg
            if new_d2p < min_pixels_per_degree:
                # this zoom is too much; we will try to zoom to the min d2p instead
                ideal_zoom = d2p / min_pixels_per_degree
                im = sktr.pyramid_reduce(imar[0], ideal_zoom, mode='constant', cval=background,
                                         multichannel=False)
                new_d2p = float(im.shape[0]) / imsz_deg
                # if this still didn't work, we aren't resizing, just using what we have
                if new_d2p < min_pixels_per_degree:
                    new_d2p = d2p
                    ideal_zoom = 1
            # okay, at this point, we've only failed to find a d2p if ideal_zoom is 1
            if ideal_zoom != 1 and new_d2p != d2p:
                # resize and update values
                imar = sktr.pyramid_reduce(imar.T, ideal_zoom,
                                           mode='constant', cval=background, multichannel=True).T
                d2p = new_d2p
                cpp = cpd / d2p
        # Okay! We have resized if needed, now we do all the filters for this spatial frequency
        if use_spatial_gabors:
            # Using convolution with Gabors
            filters = np.asarray([gabor_kernel(cpp, theta=th) for th in gabor_orientations])
            if pool is None:
                freal = np.asarray(
                    [[ndi.convolve(im, k.real, mode='constant', cval=background) for im in imar]
                     for k in filters])
                fimag = np.asarray(
                    [[ndi.convolve(im, k.imag, mode='constant', cval=background) for im in imar]
                     for k in filters])
                filt_ims = freal + 1j*fimag
            else:
                iis = np.asarray(np.round(np.linspace(0, imar.shape[0], len(pool._pool))),
                                 dtype=np.int)
                i0s = iis[:-1]
                iis = iis[1:]
                filt_ims = np.asarray(
                    [np.concatenate(
                        [x for x in pool.map(
                            _convolve_from_arg,
                            [(imar[i0:ii],k,background) for (i0,ii) in zip(i0s,iis)])
                         if len(x) > 0],
                        axis=0)
                     for k in filters])
        else:
            # Using the steerable pyramid
            filters = None
            if pool is None:
                filt_ims = np.asarray([spyr_filter(imar, th, cpp, 1, len(gabor_orientations))
                                       for th in gabor_orientations])
            else:
                iis = np.asarray(np.round(np.linspace(0, imar.shape[0], len(pool._pool))),
                                 dtype=np.int)
                i0s = iis[:-1]
                iis = iis[1:]
                filt_ims = np.asarray(
                    [np.concatenate(
                        [x for x in pool.map(
                            _spyr_from_arg,
                            [(imar[i0:ii],th,cpp,nth) for (i0,ii) in zip(i0s,iis)])
                         if len(x) > 0],
                        axis=0)
                     for th in gabor_orientations])
        # add the results to the lists of results
        filt_ims.setflags(write=False)
        imar.setflags(write=False)
        if filters is not None: filters.setflags(write=False)
        oces[cpd] = filt_ims
        d2ps[cpd] = d2p
        sims[cpd] = imar
        flts[cpd] = filters
    if pool is not None: pool.close()
    # okay, we've finished; just mark things as read-only and make lists into tuples...
    cpds = [pimms.mag(cpd, 'cycles/degree') for cpd in gabor_spatial_frequencies]
    (oces, d2ps, sims, flts) = [tuple([m[cpd] for cpd in cpds])
                                for m in (oces, d2ps, sims, flts)]
    # and return!
    return {'oriented_contrast_images': oces,
            'scaled_pixels_per_degree': d2ps,
            'scaled_image_arrays':      sims,
            'gabor_filters':            flts}

def divisively_normalize_Heeger1992(data, cpds, divisive_exponent=0.5, saturation_constant=1.0):
    '''
    divisively_normalize(data) yields the 4D image array that is the result of divisively
    normalizing the 3D orientation-filtered image array values of the array data, whose elements
    should be the contrast image array at all spatial frequencies and orientations (i.e., each
    element of the tuple data is a numpy array for a specific spatial frequency with 4 axes: axis 0
    varies by filter orientation, axis 1 varies by image, and axis 2 and 3 are image shape).
    
    Reference:
      Heeger DJ (1992) Vis. Neurosci. 9(2):181-197. doi:10.1017/S0952523800009640.
    '''
    s = saturation_constant
    r = divisive_exponent
    dens = [(s**r + np.mean(d, axis=0)**r) for d in data]
    normalized = [np.mean(d**r, axis=0) / den for (d,den) in zip(data, dens)]
    for n in normalized: n.setflags(write=False)
    return tuple(normalized)

def divisively_normalize_Heeger1992_square(data, cpds,
                                           divisive_exponent=1, saturation_constant=1.0):
    '''
    Same as "divisively_normalize_Heeger1992", but squares data before divisive normalization and
    sums across orientations for the surround term.
    '''
    if pimms.is_ndarray(data): data = data**2
    else: data = tuple([d**2 for d in data])
    return divisively_normalize_Heeger1992_square(data, cpds,
                                                  divisive_exponent=divisive_exponent,
                                                  saturation_constant=saturation_constant)

def divisively_normalize_naive(data, divisive_exponent=0.5, saturation_constant=1.0):
    '''
    divisively_normalize_naive(data) yields the 3D image array that is the result of divisively
    normalizing the 4D orientation-filtered image array values of the tuple of 3D arrays, data. The
    first axis (the tuple axis) corresponds to spatial frequencies
    '''
    tup = [np.mean(d, axis=0) for d in data]
    for d in tup: d.setflags(write=False)
    return tuple(tup)

def divisively_normalize_spatialfreq(data, cpds,
                                     background=0.5, divisive_exponent=2, saturation_constant=0.1):
    '''
    Divisively normalizes data taking into account the previous and following spatial frequency
    level. Data is the 4D decomposition of an image into spatial frequencies and orientations, such
    as the result of the steerable pyramid transform.

    Author: Chrysa Papadaniil <chrysa@nyu.edu>
    '''
    s = saturation_constant
    r = divisive_exponent
    ii = np.argsort([pimms.mag(cpd, 'cycles/degree') for cpd in cpds])
    if not np.array_equal(ii, range(len(ii))):
        cpds = cpds[ii]
        data = data[ii]
    numlevels = len(data)
    normalizers = [np.sum(d, axis=0) for d in data]
    normalized = []
    for level in range(numlevels):
        neis = [ii for ii in (level-1, level+1) if ii >=0 if i < numlevels]
        n0 = np.array(normalizers[level])
        for ii in neis:
            ni = normalizers[ii]
            if ni.shape[0] != n0.shape[0]:
                ni = np.array(ni)
                zoom = float(n0.shape[0]) / float(ni.shape[0])
                f = sktr.pyramid_expand if zoom > 1 else sktr.pyramid_reduce
                n0 += f(ni, zoom, mode='constant', cval=background, multichannel=False)
        n0 /= len(neis) # use the mean
        normalized.append(np.mean(data[level]**r, axis=0) / (s**r + n0**r))
    for nrm in normalized: nrm.setflags(write=False)
    return tuple(normalized)

@pimms.calc('contrast_constants', cache=True)
def calc_contrast_constants(labels, contrast_constants_by_label):
    '''
    calc_contrast_constants is a calculator that translates contrast_constants_by_label into a
    numpy array of contrast constants using the labels parameter.

    Required afferent parameters:
      * labels
      @ contrast_constants_by_label Must be a map whose keys are label values and whose values are
        the variance-like contrast constant for that particular area; all values appearing in the
        pRF labels must be found in this map.

    Provided efferent parameters:
      @ contrast_constants Will be an array of values, one per pRF, of the contrast constants.
    '''
    r = np.asarray(lookup_labels(labels, contrast_constants_by_label), dtype=np.dtype(float).type)
    r.setflags(write=False)
    return r

@pimms.calc('compressive_constants', cache=True)
def calc_compressive_constants(labels, compressive_constants_by_label):
    '''
    calc_compressive_constants is a calculator that translates compressive_constants_by_label into a
    numpy array of compressive constants using the labels parameter.

    Required afferent parameters:
      * labels
      @ compressive_constants_by_label Must be a map whose keys are label values and whose values
        are the compressive output exponent for that particular area; all values appearing in the
        pRF labels must be found in this map.

    Provided efferent parameters:
      @ compressive_constants Will be an array of compressive output exponents, one per pRF.
    '''
    r = np.asarray(lookup_labels(labels, compressive_constants_by_label),
                   dtype=np.dtype(float).type)
    r.setflags(write=False)
    return r

@pimms.calc('normalized_contrast_images', cache=True)
def calc_divisive_normalization(oriented_contrast_images, scaled_pixels_per_degree,
                                gabor_spatial_frequencies,
                                divisive_exponent=0.5, saturation_constant=1.0,
                                divisive_normalization_schema='Heeger1992'):
    '''
    calc_divisive_normalization is a calculator that performs divisive normalization on the
    filtered images in oriented_contrast_images. The divisive normalization performed is by default
    that described by Heeger (1992), but can be changed via the divisive_normalization_schema
    option.

    Required afferents:
      * oriented_contrast_images
      * scaled_pixels_per_degree
      * gabor_spatial_frequencies
      @ divisive_exponent Must be the power to which to raise the terms in the divisive
        normalization calculation.
      @ saturation_constant Must be the saturation constant s in the divisive normalization
        calculation.
      @ divisive_normalization_schema Must be the name of the schema to use. Currently supported
        schemas are 'Heeger1992' (the default), 'Heeger1992_square', and 'naive'.
    '''
    divisive_normalization_schema = divisive_normalization_schema.lower()
    oriented_contrast_images = tuple(map(np.abs, oriented_contrast_images))
    if divisive_normalization_schema == 'heeger1992':
        r = divisively_normalize_Heeger1992(oriented_contrast_images, gabor_spatial_frequencies,
                                            divisive_exponent=divisive_exponent,
                                            saturation_constant=saturation_constant)
    elif divisive_normalization_schema == 'heeger1992_square':
        r = divisively_normalize_Heeger1992_square(oriented_contrast_images,
                                                   gabor_spatial_frequencies,
                                                   divisive_exponent=divisive_exponent,
                                                   saturation_constant=saturation_constant)
    elif divisive_normalization_schema == 'naive':
        r = divisively_normalize_naive(oriented_contrast_images,
                                       gabor_spatial_frequencies,
                                       divisive_exponent=divisive_exponent,
                                       saturation_constant=saturation_constant)
    elif divisive_normalization_schema == 'sfreq':
        r = divisively_normalize_spatialfreq(oriented_contrast_images,
                                             gabor_spatial_frequencies,
                                             divisive_exponent=divisive_exponent,
                                             saturation_constant=saturation_constant)
    else:
        raise ValueError('unrecognized normalization schema: %s' % divisive_normalization_schema)
    return {'normalized_contrast_images': r}

# For use with multiprocessing
def _pRF_contrasts(arg):
    '''
    _pRF_contrasts actually calculates the first- and second-order contrast for a group of PRFSpec
      objects, a set of images, and the related parameters. This function requires a single tuple
      of arguments so that it is easy to use multiprocessing.Pool.map with it. The arguments are
      (pRFs, normalized_contrast_images, scaled_pixels_per_degree, contrast_constants,
       gabor_spatial_frequencies, spatial_frequency_sensitivity_function). The function returns a
      tuple of (first-order-contrast, second-order-contrast) matrices.
    '''
    (pRFs, imss, d2ps, contrast_constants, cpds, sfsf) = arg
    foc = np.zeros((len(pRFs), len(imss[0])), dtype=np.float)
    soc = np.zeros((len(pRFs), len(imss[0])), dtype=np.float)
    max_d2p = np.max(d2ps)
    imss = [(ims if d2p == max_d2p else
             sktr.pyramid_expand(ims.T, max_d2p/d2p, cval=0, mode='constant', multichannel=True).T)
        for (ims, d2p) in zip(imss, d2ps)]
    d2ps = [max_d2p for _ in d2ps]
    for (ii,prf,c) in zip(range(len(pRFs)), pRFs, contrast_constants):
        # For each pRF we first want the spatial frequency weights
        ws = np.asarray(sfsf(prf, cpds))
        ws *= ny.util.zinv(np.sum(ws))
        # Okay, now we extract the pixels and weights from each of the images
        dat = []
        for (ims, d2p, sfw) in zip(imss, d2ps, ws):
            if sfw <= 0.001: continue
            (pxs, rfw) = prf(ims, d2p)
            rfw *= ny.util.zinv(np.sum(rfw))
            dat.append((pxs, rfw*sfw))
        # Okay, now we just need to calculate first and second order contrast...
        for (pxs, w) in dat:
            tmp = np.sum(pxs * w, axis=1)
            foc[ii,:] += tmp
            tmp = np.transpose([tmp])
            soc[ii,:] += np.sum(w * (pxs - c*tmp)**2, axis=1)
    return (foc, soc)

@pimms.calc('pRF_FOC', 'pRF_SOC', cache=True)
def calc_pRF_contrasts(pRFs, normalized_contrast_images, scaled_pixels_per_degree,
                       spatial_frequency_sensitivity_function, gabor_spatial_frequencies,
                       contrast_constants, multiprocess=True):
    '''
    calc_pRF_contrasts is a calculator that is responsible for calculating both the first order
    and the second order contrasts within each pRF. The first order contrast (pRF_FOC) is just
    the weighted mean of the contrasts within each pRF. The second order contrast is the second
    moment of the contrast about (c * pRF_FOC) where c is the contrast_constant for the pRF. The
    calculations are made using normalized_contrast_images, which represents the contrast after
    divisive normalization.

    Required afferent parameters:
      * pRFs
      * normalized_contrast_images
      * scaled_pixels_per_degree
      * contrast_constants
      * gabor_spatial_frequencies
      @ spatial_frequency_sensitivity_function Must be a function of two arguments: a PRFSpec
        object and a a numpy array of spatial frequencies. The return value should be a list of
        weights, one per spatial frequency, for the particular pRF.

    Provided efferent parameters:
      @ pRF_FOC Will be an array of the first-order-contrast values, one per pRF per image;
        these will be stored in an (n x m) matrix where n is the number of pRFs and m is the
        number of images.
      @ pRF_SOC Will be an array of the second-order-contrast values, one per pRF per image;
        these will be stored in an (n x m) matrix where n is the number of pRFs and m is the
        number of images. Note that this is not the variance of the contrast exactly, but is
        the weighted second moment of the contrast about the weighted mean (pRF_FOC).
    '''
    sfsf = spatial_frequency_sensitivity_function
    if pimms.is_str(sfsf): sfsf = global_lookup(sfsf)
    gabor_spatial_frequencies = pimms.mag(gabor_spatial_frequencies, 'cycles/degree')
    # Okay, we're going to walk through the pRFs
    ok = False
    if multiprocess:
        pool = None
        try:
            import multiprocessing as mp
            foc = np.zeros((len(pRFs), len(normalized_contrast_images[0])), dtype=np.float)
            soc = np.zeros((len(pRFs), len(normalized_contrast_images[0])), dtype=np.float)
            pool = mp.Pool(mp.cpu_count() if multiprocess is True else multiprocess)
            idcs = np.array(np.round(np.linspace(0, len(pRFs), pool._processes + 1)), dtype=np.int)
            idcs = np.asarray([ii for ii in zip(idcs[:-1],idcs[1:])])
            tmp = [(pRFs[i0:ii], normalized_contrast_images, scaled_pixels_per_degree,
                    contrast_constants[i0:ii], gabor_spatial_frequencies, sfsf)
                   for (i0,ii) in idcs]
            tmp = pool.map(_pRF_contrasts, tmp)
            for ((f,s),(i0,ii)) in zip(tmp, idcs):
                foc[i0:ii] = f
                soc[i0:ii] = s
            ok = True
        except: pass
        finally:
            if pool is not None: pool.close()
    if not ok:
        (foc, soc) = _pRF_contrasts((pRFs, normalized_contrast_images, scaled_pixels_per_degree,
                                     contrast_constants, gabor_spatial_frequencies, sfsf))
    # That's all!
    foc.setflags(write=False)
    soc.setflags(write=False)
    return (foc, soc)

@pimms.calc('gains', cache=True)
def calc_gains(labels, gains_by_label):
    '''
    calc_gains is a calculator that translates gains_by_label into a numpy array of contrast
    constants using the labels parameter.

    Required afferent parameters:
      * labels
      @ gains_by_label Must be a map whose keys are label values and whose values are
        the variance-like contrast constant for that particular area; all values appearing in the
        pRF labels must be found in this map.

    Provided efferent parameters:
      @ gains Will be an array of values, one per pRF, of the gains.
    '''
    r = np.array(lookup_labels(labels, gains_by_label), dtype='float')
    r.setflags(write=False)
    return r

@pimms.calc('prediction', cache=True)
def calc_compressive_nonlinearity(pRF_SOC, compressive_constants, gains):
    '''
    calc_compressive_nonlinearity is a calculator that applies a compressive nonlinearity to the
    predicted second-order-contrast responses of the pRFs. If the compressive constant is n is and
    the SOC response is s, then the result here is simply s ** n.

    Required afferent parameters:
      * pRF_SOC
      * compressive_constants
      * gains

    Provided efferent values:
      @ prediction Will be the final predictions of %BOLD-change for each pRF examined, up to gain.
        The data will be stored in an (n x m) matrix where n is the number of pRFs (see labels,
        hemispheres, cortex_indices) and m is the number of images.
    '''
    gains = np.reshape(gains, (-1,1))
    out = gains * np.asarray([s**n for (s, n) in zip(pRF_SOC, compressive_constants)])
    out.setflags(write=False)
    return out
    
