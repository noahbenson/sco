####################################################################################################
# stimulus/core.py
# The stimulus module of the standard cortical observer; core definitions and checks.
# By Noah C. Benson

import numpy                 as     np
import pyrsistent            as     pyr

import pimms, os, sys, warnings

from   ..util                import (lookup_labels, units)

from   scipy                 import ndimage    as ndi
from   scipy.interpolate     import (RectBivariateSpline, interp1d)

from   skimage               import data
from   skimage.util          import img_as_float

warnings.filterwarnings('ignore', category=UserWarning, message='.*From scipy 0.13.0.*')

@pimms.calc('gamma_correction_function')
def calc_gamma_correction(gamma=None):
    '''
    calc_gamma_correction is a calculator that accepts an optional argument gamma and provides a
    value gamma_correction_function that corrects the contrast of a stimulus presentation.

    Optional afferent values:
      @ gamma May be given, in which case it must be one of:
        - an (n x 2) or (2 x n) matrix such that is equivalent to (potentially after transposition)
          a matrix of (x,y) values where x is the input gamma and y is the corrected gamma
        - a vector of corrected gamma values; if the vector u is of length n, then this is 
          equivalent to passing a matrix in which the y-values are the elements of u and the
          x-values are evenly spaced values that cover the interval [0,1]; accordingly there must be
          at least 2 elements
        - a function that accepts a number between 0 and 1 and returns the corrected gamma
        By default this is None, and no gamma correction is applied.
    '''
    # First, setup the stimulus_gamma correction:
    if gamma is None:                return lambda x: x
    elif hasattr(gamma, '__call__'): return gamma
    elif hasattr(gamma, '__iter__'):
        vals = np.array(gamma)
        if len(vals.shape) > 2:
            raise ValueError('stimulus_gamma must be 1D or 2D array')
        if len(vals.shape) == 1:
            n = float(vals.shape[0] - 1)
            vals = np.asarray([[float(i)/n for i in range(vals.shape[0])], vals])
        # Okay, assume here that vals is nx2 or 2xn
        if vals.shape[1] != 2: vals = vals.T
        # and interpolate these
        return interp1d(vals[:,0], vals[:,1], kind='cubic')
    else:
        raise ValueError('Given stimulus_gamma argument has neither iter nor call attribute')

def import_stimulus(stim, gcf):
    '''
    import_stimulus(stim, gcf) yields the imported image for the given stimulus argument stim; stim
    may be either a filename or an image array; the argument gcf must be the gamma correction
    function.
    '''
    if isinstance(stim, basestring):
        im = np.asarray(data.load(stim), dtype=np.float)
    else:
        im = np.asarray(stim, dtype=np.float)
    if len(im.shape) == 3:
        # average the color channels
        im = np.mean(im, axis=2)
    if len(im.shape) != 2:
        raise ValueError('images must be 2D or 3D matrices')
    # We need to make sure this image is between 0 and 1; if not, we assume it's between 0 and 255;
    # for now it seems safe to automatically detect this
    mx = np.max(im)
    if not np.isclose(mx, 1) and mx > 1: im = im/255.0
    # if we were given a color image,
    if gcf is not None: im = gcf(im)
    return im

@pimms.calc('stimulus_map', 'stimulus_ordering', cache=True)
def import_stimuli(stimulus, gamma_correction_function):
    '''
    import_stimuli is a calculation that ensures that the stimulus images to be used in the sco
    calculation are properly imported.

    Required afferent values:
      @ stimulus May either be a dict or list of images matrices or a list of image filenames.

    Optional afferent values:
      @ gamma_correction_function May specifies how gamma should be corrected; this
        should usually be provided via the gamma argument (see calc_gamma_correction and gamma).

    Efferent output values:
      @ stimulus_map Will be a persistent dict whose keys are the image identifiers and whose
        values are the image matrices of the imported stimuli prior to normalization or any
        processing.
      @ stimulus_ordering Will be a persistent vector of the keys of stimulus_map in the order
        provided.
    '''
    # Make this into a map so that we have ids and images/filenames
    if not pimms.is_map(stimulus):
        # first task: turn this into a map
        if isinstance(stimulus, basestring):
            stimulus = {stimulus: stimulus}
            order = [stimulus]
        elif hasattr(stimulus, '__iter__'):
            pat = '%%0%dd' % (int(np.log10(len(stimulus))) + 1)
            order = [(pat % i) for i in range(len(stimulus))]
            stimulus = {(pat % i):s for (i,s) in enumerate(stimulus)}
        else:
            raise ValueError('stimulus is not iterable nor a filename')
    else:
        order = stimulus.keys()
    # we can use the stimulus_importer function no matter what the stimulus arguments are
    stim_map = {k:import_stimulus(v, gamma_correction_function) for (k,v) in stimulus.iteritems()}
    for u in stim_map.itervalues():
        u.setflags(write=False)
    return {'stimulus_map': pyr.pmap(stim_map), 'stimulus_ordering': pyr.pvector(order)}

def image_apply_aperture(im, radius,
                         center=None, fill_value=0.5, edge_width=10, crop=True):
    '''
    image_apply_aperture(im, rad) yields an image that has been given a circular aperture centered
    at the middle of the image im with the given radius rad in pixels. The following options may be
    given:
      * fill_value (default 0.5) gives the value filled in outside of the aperture
      * crop (default True) indicates whether the image should be cropped after the aperture is
        applied; possible values are a tuple (r,c) indicating the desired size of the resulting
        image; an integer r, equivalent to (r,r); or True, in which case the image is cropped such
        that it just holds the entire aperture (including any smoothed edge).
      * edge_width (default 10) gives the number of pixels over which the aperture should be
        smoothly extended; 0 gives a hard edge, otherwise a half-cosine smoothing is used.
      * center (default None) gives the center of the aperture as a (row, column) value; None uses
        the center of the image.
    '''
    im = np.asarray(im)
    # First, figure out the final image size
    crop      = 2*radius if crop is True else crop
    final_sz  = crop if isinstance(crop, (tuple, list)) else (crop, crop)
    final_sz  = [int(round(x)) for x in final_sz]
    final_im  = np.full(final_sz, fill_value)
    # figure out the centers
    center       = (0.5*im.shape[0],       0.5*im.shape[1])       if center is None else center
    final_center = (0.5*final_im.shape[0], 0.5*final_im.shape[1])
    # we may have to interpolate pixels, so setup the interpolation; (0,0) in the lower-left:
    interp = RectBivariateSpline(range(im.shape[0]), range(im.shape[1]), im)
    # prepare to interpolate: find the row/col values for the pixels into which we interpolate
    rad2 = radius**2
    final_xy = [(x,y)
                for x in range(final_im.shape[0]) for xx in [(x - final_center[0])**2]
                for y in range(final_im.shape[1]) for yy in [(y - final_center[1])**2]
                if xx + yy <= rad2]
    f2i = float(2 * radius) / float(final_sz[0])
    image_xy = [(x,y)
                for xy in final_xy
                for (dx,dy) in [(xy[0] - final_center[0], xy[1] - final_center[1])]
                for (x,y) in [(dx*f2i + center[0], dy*f2i + center[1])]]
    final_xy = np.transpose(final_xy)
    image_xy = np.transpose(image_xy)
    # pull the interpolated values out of the interp structure:
    z = interp(image_xy[0], image_xy[1], grid=False)
    # and put these values into the final image
    for ((x,y),z) in zip(final_xy.T, z): final_im[x,y] = z
    # now, take care of the edge
    if edge_width is 0: return final_im
    erad2 = (radius - edge_width)**2
    for r in range(final_im.shape[0]):
        for c in range(final_im.shape[1]):
            r0 = float(r) - final_center[0]
            c0 = float(c) - final_center[1]
            d0 = r0*r0 + c0*c0
            if d0 > erad2 and d0 <= rad2:
                d0 = np.sqrt(d0) - radius + edge_width
                w = 0.5*(1.0 + np.cos(d0 * np.pi / edge_width))
                final_im[r,c] = w*final_im[r,c] + (1.0 - w)*fill_value
    # That's it!
    return final_im

@pimms.calc('image_array', 'image_names', 'pixel_centers', cache=True)
def calc_images(pixels_per_degree, stimulus_map, stimulus_ordering,
                background=0.5,
                aperture_radius=None,
                aperture_edge_width=None,
                normalized_pixels_per_degree=None):
    '''
    calc_images() is a the calculation that converts the imported_stimuli value into the normalized
    images value.
    
    Required afferent parameters:
      @ pixels_per_degree Must specify the number of pixels per degree in the input images; note
        that all stimulus images must have the same pixels_per_degree value.
      @ stimulus_map Must be a map whose values are 2D image matrices (see import_stimuli).
      @ stimulus_ordering Must be a list of the stimulus filenames or IDs (used by calc_images to
        ensure the ordering of the resulting image_array datum is correct; see also import_stimuli).

    Optional afferent parameters:
      @ background Specifies the background color of the stimulus; by default this is 0.5 (gray);
        this is only used if an aperture is applied.
      @ aperture_radius Specifies the radius of the aperture in degrees; by default this is None,
        indicating that no aperture should be used; otherwise the aperture is applied after
        normalizing the images.
      @ aperture_edge_width Specifies the width of the aperture edge in degrees; by default this is
        None; if 0 or None, then no aperture edge is used.
      @ normalized_pixels_per_degree Specifies the resolution of the images used in the calculation;
        by default this is the same as pixels_per_degree.

    Output efferent values:
      @ image_array Will be the 3D numpy array image stack; image_array[i,j,k] is the pixel in image
        i, row j, column k
      @ image_names Will be the list of image names in the same order as the images in image_array;
        the names are derived from the keys of the stimulus_map.
      @ pixel_centers Will be an r x c x 2 numpy matrix with units of degrees specifying the center
        of each pixel (r is the number of rows and c is the number of columns).
    '''
    # first, let's interpret our arguments
    deg2px = float(pimms.mag(pixels_per_degree, 'px/deg'))
    if normalized_pixels_per_degree is None:
        normdeg2px = deg2px
    else:
        normdeg2px = float(pimms.mag(normalized_pixels_per_degree, 'px/deg'))
    # we can get the zoom ratio from these
    zoom_ratio = normdeg2px / deg2px
    # Zoom each image so that the pixels per degree is right:
    if np.isclose(zoom_ratio, 1):
        imgs = stimulus_map
    else:
        imgs = {k:ndi.zoom(im, zoom_ratio, cval=background) for (k,im) in stimulus_map.iteritems()}
    maxdims = [np.max([im.shape[i] for im in imgs.itervalues()]) for i in [0,1]]
    # Then apply the aperture
    if aperture_radius is None:
        aperture_radius = (0.5 * np.sqrt(np.dot(maxdims, maxdims))) / normdeg2px
    if aperture_edge_width is None:
        aperture_edge_width = 0
    rad_px = 0
    try: rad_px = pimms.mag(aperture_radius, 'deg') * normdeg2px
    except:
        try: rad_px = pimms.mag(aperture_radius, 'px')
        except: raise ValuerError('aperture_radius given in unrecognized units')
    aew_px = 0
    try: aew_px = pimms.mag(aperture_edge_width, 'deg') * normdeg2px
    except:
        try: aew_px = pimms.mag(aperture_edge_width, 'px')
        except: raise ValuerError('aperture_edge_width given in unrecognized units')
    bg = background
    imgs = {k:image_apply_aperture(im, rad_px, fill_value=bg, edge_width=aew_px)
            for (k,im) in imgs.iteritems()}
    # Separate out the images and names and
    imar = np.asarray([imgs[k] for k in stimulus_ordering], dtype=np.float)
    imar.setflags(write=False)
    imnm = pyr.pvector(stimulus_ordering)
    # Finally, note the pixel centers
    (rs,cs) = (imar.shape[1], imar.shape[2])
    x0 = (0.5*rs, 0.5*cs)
    (r0s, c0s) = [(np.asarray(range(u)) - 0.5*u + 0.5) / deg2px for u in [rs,cs]]
    pxcs = np.asarray([[(c,-r) for c in c0s] for r in r0s], dtype=np.float)
    pxcs.setflags(write=False)
    return {'image_array': imar, 'image_names': imnm, 'pixel_centers': pxcs}
