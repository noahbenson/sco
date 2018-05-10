####################################################################################################
# contrast/core.py
# The second-order contrast module of the standard cortical observer.
# By Noah C. Benson

import numpy                 as     np
import numpy.matlib          as     npml
import pyrsistent            as     pyr
from   scipy                 import ndimage as ndi
from   skimage.filters       import gabor_kernel
from   types                 import (IntType, LongType)
from   ..util                import (lookup_labels, units, global_lookup)
from   .spyr                 import spyr_filter

import pimms

def scaled_gabor_kernel(cpp, theta=0, zero_mean=True, **kwargs):
    '''
    scaled_gabor_kernel(...) is identical to gabor_kernel(...) except that the resulting kernel is
    scaled such that the response of the kernel to a grating of identical frequency and angle and
    min/max values of -/+ 1 is 1.
    scaled_gabor_kernel has one additional argument, zero_mean=True, which specifies that the kernel
    should (or should not) be given a zero mean value. The gabor_kernel function alone does not do
    this, but scaled_gabor_kernel does this by default, unless zero_mean is set to False.
    '''
    if pimms.is_quantity(cpp): cpp = cpp.to(units.cycle / units.px).m
    if pimms.is_quantity(theta): theta = theta.to(units.rad).m
    kern = gabor_kernel(cpp, theta=theta, **kwargs)
    # First, zero-mean them
    if zero_mean:
        kern = (kern.real - np.mean(kern.real)) + 1j*(kern.imag - np.mean(kern.imag))
    # Next, make the max response grating
    (n,m) = kern.shape
    (cn,cm) = [0.5*(q - 1) for q in [n,m]]
    (costh, sinth) = (np.cos(theta), np.sin(theta))
    mtx = (2*np.pi*cpp) * np.asarray([[costh*(col - cm) + sinth*(row - cn) for col in range(m)]
                                      for row in range(n)])
    re = kern.real / np.sum(np.abs(kern.real * np.cos(mtx)))
    im = kern.imag / np.sum(np.abs(kern.imag * np.sin(mtx)))
    return np.asarray(re + 1j*im)

@pimms.immutable
class ImageArrayContrastFilter(object):
    '''
    The ImageArrayContrastFilter class is a immutable class (using pimms) that is initialized with
    a stack of images and various meta-data (see calc_contrast_images function); once initialized,
    an object f can be called (f(cpd)) to produce a map whose keys are orientations (in radians) and
    whose values are stacks of equivalent images filtered at the given frequency cpd.
    '''

    def __init__(self, image_array, d2p, gabor_orientations, background,
                 spatial_gabors=False, bandwidth=None):
        self.image_array = image_array
        self.pixels_per_degree = d2p
        self.gabor_orientations = gabor_orientations
        self.background = background
        self.spatial_gabors = spatial_gabors
        self.bandwidth = bandwidth
    def __hash__(self):
        return pimms.qhash((self.image_array,
                            self.pixels_per_degree,
                            self.gabor_orientations,
                            self.background))
    @pimms.param
    def spatial_gabors(sg):
        '''
        f.spatial_gabors is either True or False indicating whether the given 
        ImageArrayContrastFilter object f is using spatial gabors (True) or the steerable pyramid
        (False).
        '''
        return True if sg else False
    @pimms.param
    def bandwidth(bw):
        '''
        f.bandwidth is either None if using spatial_gabors (f.spatial_gabors is True); otherwise,
        is the bandwidth of each spatial frequency band used by the steerable pyramid.
        '''
        if not pimms.is_real(bw) or bw <= 0:
            raise ValueError('bandwidth must be a positive real number')
        return bw
    @pimms.param
    def image_array(ims):
        '''
        f.image_array is the (numpy array) image stack tracked by the given ImageArrayContrastFilter
        object f. The image array should always be a 3D numpy array.
        '''
        if not isinstance(ims, np.ndarray):
            ims = np.array(ims, dtype=np.dtype(float).type)
            ims.setflags(write=False)
        elif not np.issubdtype(ims.dtype, np.dtype(float).type) or not ims.flags['WRITEABLE']:
            ims = np.array(ims, dtype=np.dtype(float).type)
            ims.setflags(write=False)
        if len(ims.shape) != 3:
            raise ValueError('image_array must be a stack of 2D images')
        return ims
    @pimms.param
    def pixels_per_degree(d2p):
        '''
        f.pixels_per_degree is the number of pixels per degree in the visual field of the image
        array tracked by the ImageArrayContrastFilter object f.
        '''
        d2p_unit = units.px / units.degree
        d2p = d2p.to(d2p_unit) if pimms.is_quantity(d2p) else d2p * d2p_unit
        return d2p
    @pimms.param
    def gabor_orientations(go):
        '''
        f.gabor_orientations is a read-only numpy array of the gabor orientations at which to
        examine contrast; all elements are in radians.
        '''
        if not hasattr(go, '__iter__'):
            go = np.asarray(range(go), dtype=np.dtype(float).type)
            go *= np.pi / float(len(go))
        urad = units.rad
        go = urad * np.asarray([g.to(urad).m if pimms.is_quantity(g) else g for g in go],
                               dtype=np.dtype(float).type)
        go.setflags(write=False)
        return go
    @pimms.param
    def background(bg):
        '''
        f.background is the contrast value (between 0 and 1) of the background that is used when
        a filter extends beyond the range of the image. Usually this is 0.5 (gray).
        '''
        bg = float(bg)
        if bg < 0 or bg > 1: raise ValueError('background must be in the range [0,1]')
        return bg
    def to_cycles_per_pixel(self, cpd):
        '''
        f.to_cycles_per_pixel(cpd) yields the given cpd value in cycles per pixel; if cpd has no
        units it is assumed to be in units of cycles per degree. This uses the conversion factor
        stored in f.pixels_per_degree to make any required conversion.
        '''
        cpp = None
        if pimms.is_quantity(cpd):
            try:    cpp = cpd.to(units.cycles / units.degree) / self.pixels_per_degree
            except: cpp = None
            if cpp is None:
                try:    cpp = cpd.to(units.cycles / units.pixel)
                except: raise ValueError('frequency must be in cycles/degree or cycles/pixel')
        else:
            cpp = (cpd * (units.cycles / units.degree)) / self.pixels_per_degree
        # also, cast to float...
        cpp = float(cpp.m) * cpp.u
        return cpp
    def __call__(self, cpd, bandwidth=1):
        '''
        f(cpd) yields an image stack identical in shape to f.image_array which has been filtered
        at the given frequency cpd (in cycles per degree). The cpd argument may be in different
        units as long as the units are annotated using pint. Valid units are 
        {cycles/radians/degrees} per {degrees/radians/cycles, pixels}.
        '''
        cpp = self.to_cycles_per_pixel(cpd)
        # we need to calculate a new set of filtered images...
        # The filtered orientations
        res = {}
        bg = self.background
        for th in self.gabor_orientations:
            if self.spatial_gabors:
                kerns = scaled_gabor_kernel(cpp.m, theta=pimms.mag(th, 'rad'))
                kerns = (kerns.real, kerns.imag)
                rmtx = np.zeros(self.image_array.shape)
                for (i,im) in enumerate(self.image_array):
                    rmtx[i, :, :] = np.sqrt(
                        np.sum([ndi.convolve(im, k, mode='constant', cval=self.background)**2
                                for k in kerns],
                               axis=0))
            else:
                rmtx = np.sqrt(spyr_filter(self.image_array, pimms.mag(th, 'rad'),
                                           cpd, bandwidth,
                                           len(self.gabor_orientations)))
            rmtx.setflags(write=False)
            res[th] = rmtx
        return pyr.pmap(res)

@pimms.immutable
class ImageArrayContrastView(object):
    '''
    The ImageArrayContrastView class is an immutable (via pimms) data structure that calculates and
    stores the first-order contrast-base responses to an image array given a set of parameters
    governing its divisive normalization scheme and a divisive normalization function itself. The
    main purpose of this class is to handle caching of individual results and allow the divisive
    normalization function definition itself to be very clean.
    '''
    def __init__(self, contrast_filter, freq, divnorm_fn, params):
        self.contrast_filter = contrast_filter
        self.frequency = freq
        self.parameters = params
        self.divisive_normalization_fn = divnorm_fn
    def __hash__(self):
        return hash((self.contrast_filter, self.frequency, self.parameters,
                     self.divisive_normalization_fn))
    @pimms.param
    def frequency(cpd):
        '''
        rsp.frequency is the frequency at which the image array response results are calculated.
        This may be in cycles per degree or cycles per pixel; alternately use rsp.cpd or rsp.cpp.
        '''
        if pimms.is_quantity(cpd):
            org = cpd
            try:    cpd = org.to(units.cycles / units.degree)
            except: cpd = None
            if cpd is None:
                try:    cpd = org.to(units.cycles / units.pixel)
                except: raise ValueError('frequency must be in cycles/degree or cycles/pixel')
        else:
            cpd = cpd * (units.cycles / units.degree)
        # also, cast to float...
        return float(cpd.m) * cpd.u
    @pimms.param
    def contrast_filter(f):
        '''
        rsp.contrast_filter is a ImageArrayContrastFilter object that stores the image_array over
        which the object rsp calculates.
        '''
        if not isinstance(f, ImageArrayContrastFilter):
            raise ValueError('contrast_filter must be an ImageArrayContrastFilter object')
        return f
    @pimms.value
    def cpd(frequency, contrast_filter):
        '''
        rsp.cpd is the frequency in cycles-per-degree at which the image array response results are
        calculated in the object rsp.
        '''
        ppd = contrast_filter.pixels_per_degree
        return frequency if frequency.u == units.cycles/units.degree else frequency * ppd
    @pimms.value
    def cpp(frequency, contrast_filter):
        '''
        rsp.cpp is the frequency in cycles-per-pixel at which the image array response results are
        calculated in the object rsp.
        '''
        ppd = contrast_filter.pixels_per_degree
        return frequency if frequency.u == units.cycles/units.px else frequency / ppd
    @pimms.param
    def parameters(params):
        '''
        rsp.parameters is a pimms.itable object of the divisive normalization parameters (each row
        of the itable is one parameterization) used in the calculation for the immage array response
        object rsp. The parameters may be given to a response object as a list of maps or as an
        itable.
        '''
        if pimms.is_itable(params):
            return params
        elif pimms.is_map(params):
            tbl = pimms.itable(params)
            # we want this to fail if it can't be transformed to rows
            try: tbl.rows
            except: raise ValueError('map could not be cast to itable')
            return tbl
        else:
            tbl = {}
            try:
                p0 = params[0]
                tbl = {k:[v] for (k,v) in p0.iteritems()}
                for p in params[1:]:
                    if len(p) != len(p0): raise ValueError()
                    for (k,v) in p.iteritems(): tbl[k].append(v)
                tbl = pimms.itable(tbl)
                tbl.rows
            except:
                raise ValueError('parameters must be an itable, a map of columns, or a list of '
                                 + 'parameter maps')
            return tbl
    @pimms.param
    def divisive_normalization_fn(divnorm_fn):
        '''
        rsp.divisive_normalization_fn is the divisive normalization function that is used with the
        given image array response object rsp. The function should accept exactly two arguments:
         (1) a mapping of orientations (in radians) to 3D numpy arrays of images filtered at the
             given orientation, and
         (2) a mapping of parameter names to values for use in the normalization, which are passed
             to the function via the ** operator--i.e., divisive_normalization_fn(omap, **params);
        the function should return an array of contrast energy images.
        '''
        # currently no good checks...
        if isinstance(divnorm_fn, basestring):
            # check that we can look it up
            try:
                f = global_lookup(divnorm_fn)
            except:
                raise ValueError('divnorm_fn string must be the name of a function')
            else:
                if not hasattr(f, '__call__'): raise ValueError('divnorm_fn must be callable')
        elif not hasattr(divnorm_fn, '__call__'): raise ValueError('divnorm_fn must be callable')
        return divnorm_fn
    @pimms.value
    def contrast_energy(contrast_filter, parameters, frequency, divisive_normalization_fn):
        '''
        rsp.contrast_energy is a 3D array of divisively-normalized contrast energy at the frequency
        given by rsp.frequency using the divisive normalization parameters in rsp.parameters. These
        data are stored in an n x r x c matrix where r x c is the image size and n is the number of
        images in the image array stored by the rsp.contrast_filter object.
        '''
        # see if we need to lookup the divnorm fn
        if isinstance(divisive_normalization_fn, basestring):
            dnfn = global_lookup(divisive_normalization_fn)
        # we'll build this up with unique parameterizations
        res = {}
        # get the images filtered at the appropriate frequency
        filt = contrast_filter(frequency)
        # call the divisive normalization for each parameter... cache results as we go...
        for p in parameters.rows:
            if p not in res:
                res[p] = dnfn(filt, **p)
        return pyr.pmap(res)
    
@pimms.calc('contrast_filter', memoize=True)
def calc_contrast_filter(image_array, pixels_per_degree, normalized_pixels_per_degree,
                         gabor_orientations, background,
                         cpd_sensitivities, use_spatial_gabors=False):
    '''
    calc_contrast is a calculator that takes as input a normalized image stack and various parameter
    data and produces an object of type ImageArrayContrastFilter, contrast_filterm an object that
    can be called as contrast_filter(frequency) to yield a map whose keys are gabor orientations (in
    radians) and whose values are image stacks with identical shapes as image_array but that have
    been filtered at the key's orientation and the frequency.

    Required afferent values:
      * image_array
      * normalized_pixels_per_degree
      * gabor_orientations
      * background
      @ use_spatial_gabors Must be either True (use spatial gabor filters instead of the steerable
        pyramid) or False (use the steerable pyramid); by default this is False.

    Efferent output values:
      @ contrast_filter Will be an object of type ImageArrayContrastFilter that can filter the image
        array at arbitrary frequencies and divisive normalization parameters.
    '''
    if normalized_pixels_per_degree is None: normalized_pixels_per_degree = pixels_per_degree
    all_cpds = np.unique([k.to(units.cycle/units.deg).m if pimms.is_quantity(k) else k
                          for s in cpd_sensitivities for k in s.iterkeys()])
    # find this difference...
    bw = np.mean(np.abs(all_cpds[1:] - all_cpds[:-1]) / all_cpds[:-1])
    # all the parameter checking and transformation is handled in this class
    return ImageArrayContrastFilter(image_array,         normalized_pixels_per_degree,
                                    gabor_orientations,  background,
                                    spatial_gabors=use_spatial_gabors,
                                    bandwidth=bw)

@pimms.calc('contrast_energies', cache=True)
def calc_contrast_energies(contrast_filter,
                           divisive_normalization_function, divisive_normalization_parameters,
                           cpd_sensitivities):
    '''
    calc_contrast_energies is a calculator that performs divisive normalization on the filtered
    contrast images and yields a nested map of contrast energy arrays; contrast_energies map has
    keys that are spatial frequencies (in cycles per degree) and whose values are maps; these maps
    have keys that are parameter value maps and whose values are the 3D contrast energy arrays.

    Required afferent parameters:
      * contrast_filter
      * divisive_normalization_function, divisive_normalization_parameters
      * cpd_sensitivities

    Output efferent values:
      @ contrast_energies Will be a nested map whose first level of keys are persistent-maps of the
        divisive normalization parameters and whose second level of keys are a set of frequencies;
        the values at the second level are the stacks of contrast energy images for the particular
        divisive normalization parameters and frequencies specified in the keys.
    '''
    # first, calculate the contrast energies at each frequency for all images then we combine them;
    # since this reuses images internally when the parameters are the same, it shouldn't be too
    # inefficient:
    divnfn = divisive_normalization_function
    params = divisive_normalization_parameters
    all_cpds = np.unique([k.to(units.cycle/units.deg).m if pimms.is_quantity(k) else k
                          for s in cpd_sensitivities for k in s.iterkeys()])
    all_cpds = all_cpds * (units.cycles / units.degree)
    rsps = {cpd:vw.contrast_energy
            for cpd in all_cpds
            for vw in [ImageArrayContrastView(contrast_filter, cpd, divnfn, params)]}
    # flip this around...
    flip = {}
    for (k0,v0) in rsps.iteritems():
        for (k1,v1) in v0.iteritems():
            if k1 not in flip: flip[k1] = {}
            flip[k1][k0] = v1
    rsps = pyr.pmap({k:pyr.pmap(v) for (k,v) in flip.iteritems()})
    return {'contrast_energies': rsps}

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

@pimms.calc('pRF_SOC', cache=True)
def calc_pRF_SOC(pRFs, contrast_energies, cpd_sensitivities,
                 divisive_normalization_parameters, contrast_constants,
                 pixels_per_degree, normalized_pixels_per_degree):
    '''
    calc_pRF_SOC is a calculator that is responsible for calculating the individual SOC responses
    of the pRFs by extracting their pRFs from the contrast_energies and weighting them according
    to the cpd_sensitivities.

    Required afferent parameters:
      * pRFS
      * contrast_energies
      * cpd_sensitivities
      * divisive_normalization_parameters

    Provided efferent parameters:
      @ pRF_SOC Will be an array of the second-order-contrast energies, one per pRF per image;
        these will be stored in an (n x m) matrix where n is the number of pRFs and m is the
        number of images.
    '''
    if normalized_pixels_per_degree is None: normalized_pixels_per_degree = pixels_per_degree
    d2p = normalized_pixels_per_degree
    d2p = d2p.to(units.px/units.deg) if pimms.is_quantity(d2p) else d2p*(units.px/units.deg)
    params = divisive_normalization_parameters
    n = len(next(next(contrast_energies.itervalues()).itervalues()))
    m = len(pRFs)
    imshape = next(next(contrast_energies.itervalues()).itervalues()).shape[1:3]
    imlen = imshape[0] * imshape[1]
    socs = np.zeros((m, n))
    # we want to avoid numerical mismatch, so we round the keys to the nearest 10^-5
    def chop(x):
        x = float(x.to(units.cycle/units.degree).m if pimms.is_quantity(x) else x)
        return np.round(x, 5)
    contrast_energies = {k0:{chop(k):v for (k,v) in v0.iteritems()}
                         for (k0,v0) in contrast_energies.iteritems()}
    for (i,prf,p,ss,c) in zip(range(m), pRFs, params.rows, cpd_sensitivities, contrast_constants):
        wts = None
        uu = None
        for (cpd, w) in ss.iteritems():
            cpd = chop(cpd)
            if wts is None:
                (u,wts) = prf(contrast_energies[p][cpd], d2p, c=None)
                uu = np.zeros(u.shape)
            else:
                u = prf(contrast_energies[p][cpd], d2p, c=None, weights=False)[0]
            uu += w * u
        # Here is the SOC formula: (x - c<x>)^2
        wts = npml.repmat(wts, len(uu), 1)
        mu = np.sum(wts * uu, axis=1)
        socs[i,:] = np.sum(wts * (uu - c*npml.repmat(mu, uu.shape[1], 1).T)**2, axis=1)
    socs.setflags(write=False)
    return socs

@pimms.calc('compressive_constants', cache=True)
def calc_compressive_constants(labels, compressive_constants_by_label):
    '''
    calc_compressive_constants is a calculator that translates compressive_constants_by_label into a
    numpy array of compressive constants using the labels parameter.

    Required afferent parameters:
      * labels
      @ compressive_constants_by_label Must be a map whose keys are label values and whose values are
        the compressive output exponent for that particular area; all values appearing in the
        pRF labels must be found in this map.

    Provided efferent parameters:
      @ compressive_constants Will be an array of compressive output exponents, one per pRF.
    '''
    r = np.asarray(lookup_labels(labels, compressive_constants_by_label), dtype=np.dtype(float).type)
    r.setflags(write=False)
    return r

@pimms.calc('prediction', cache=True)
def calc_compressive_nonlinearity(pRF_SOC, compressive_constants):
    '''
    calc_compressive_nonlinearity is a calculator that applies a compressive nonlinearity to the
    predicted second-order-contrast responses of the pRFs. If the compressive constant is n is and
    the SOC response is s, then the result here is simply s ** n.

    Required afferent parameters:
      * pRF_SOC
      * compressive_constants

    Provided efferent values:
      @ prediction Will be the final predictions of %BOLD-change for each pRF examined, up to gain.
        The data will be stored in an (n x m) matrix where n is the number of pRFs (see labels,
        hemispheres, cortex_indices) and m is the number of images.
    '''
    out = np.asarray([s**n for (s, n) in zip(pRF_SOC, compressive_constants)])
    out.setflags(write=False)
    return out
    
