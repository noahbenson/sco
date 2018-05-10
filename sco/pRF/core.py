####################################################################################################
# sco/pRF/core.py
# pRF-related calculations for the standard cortical observer library.
# By Noah C. Benson

import numpy                 as     np
import scipy.sparse          as     sparse
import pyrsistent            as     pyr
from   sco.util              import (units, lookup_labels, global_lookup)
import pimms, warnings

@pimms.immutable
class PRFSpec(object):
    '''
    The PRFSpec class specifies a pRF size and location such that it can be used to extract a pRF
    from an image. Generally the class is used by a combination of the matrix() method and the
    __call__ method.
    '''
    def __init__(self, center, sigma, expt, n_radii=3):
        self.center = center
        self.sigma = sigma
        self.exponent = expt
        self.n_radii = n_radii
    def __getstate__(self):
        d = self.__dict__.copy()
        d['center'] = tuple(self.center.m)
        d['sigma'] = self.sigma.m
        return d
    def __setstate__(self, d):
        object.__setattr__(self, '__dict__', d)
        c = np.asarray(d['center']) * units.deg
        c.setflags(write=False)
        object.__setattr__(self, 'center', c)
        object.__setattr__(self, 'sigma', d['sigma'] * units.deg)
        return self
    @pimms.param
    def center(pt):
        '''
        prf.center is the (x,y) coordinate vector of the center of the pRF in degrees.
        '''
        if pimms.is_quantity(pt):
            if pt.u == units.rad:   pt = pt.to(units.deg)
            elif not (pt.u == units.deg):
                raise ValueError('pRF centers must be in degrees or radians')
        else:
            # assume degrees
            pt = pt * units.deg
        pt = np.asarray(pt.m) * pt.u
        pt.setflags(write=False)
        if len(pt.shape) != 1 or  pt.shape[0] != 2:
            raise ValueError('pRF centers must be 2D')
        return pt
    @pimms.param
    def sigma(sig):
        '''
        prf.sigma is the pRF sigma parameter in degrees; see also radius.
        '''
        if pimms.is_quantity(sig):
            if sig.u == units.rad:   sig = sig.to(units.deg)
            elif not (sig.u == units.deg):
                raise ValueError('pRF sigma must be in degrees or radians')
        else:
            sig = sig * units.deg
        if sig <= 0: raise ValueError('sigma must be positive')
        return sig
    @pimms.param
    def exponent(e):
        '''
        prf.exponent is the exponent in the pRF equation; see also radius.
        '''
        if e <= 0: raise ValueError('exponent must be postive')
        return e
    @pimms.value
    def radius(sigma, exponent):
        '''
        prf.radius is the pRF radius of the given PRFSpec object; the radius of the pRF is computed
        from the sigma and exponent parameters like so:
           radius = sigma * sqrt(exponent).
        This can be observed directly from the pRF equation:
           (q: sensitivity, x: distance from pRF center, s: sigma, n: exponent, r: radius)
           q = exp(-(x/r)^2 / 2)^n
             = exp(-n (x/r)^2 / 2)
             = exp(-(sqrt(n) x/r)^2 / s)
             = exp(-(x / (r/sqrt(n)))^2 / 2)
             = exp(-(x / s)^2 / 2)
           this tells us s == r / sqrt(n) so r = s * sqrt(n)
        '''
        return sigma.to(units.deg) * np.sqrt(exponent)
    @pimms.param
    def n_radii(nr):
        '''
        prf.n_radii is a parameter that affects how many radius's worth of distance in an image the
        prf grabs when performing sums over its sensitivity field.
        '''
        if nr <= 0: raise ValueError('n_radii must be positive')
        return nr

    def _params(self, imshape, d2p):
        if len(imshape) > 2: imshape = imshape[-2:]
        if not pimms.is_quantity(d2p): d2p = d2p * (units.px/units.deg)
        imshape = imshape * units.px
        x0 = np.asarray([(imshape[0]*0.5 - self.center[1]*d2p).to(units.px).m,
                         (imshape[1]*0.5 + self.center[0]*d2p).to(units.px).m])
        rad = self.radius * d2p
        dst = (self.n_radii * rad).m
        rrng0 = (int(np.floor(x0[0] - dst)), int(np.ceil(x0[0] + dst)))
        crng0 = (int(np.floor(x0[1] - dst)), int(np.ceil(x0[1] + dst)))
        rrng = (max([rrng0[0], 0]), min([rrng0[1], imshape[0].m]))
        crng = (max([crng0[0], 0]), min([crng0[1], imshape[1].m]))
        if any(s[1] - s[0] <= 0 for s in [rrng, crng]):
            print (self.center, self.sigma, self.exponent, self.n_radii)
            print (imshape, d2p, dst, rrng0, rrng, crng0, crng)
            raise ValueError('Bad image or std given to PRFSpec._params()')
        return (x0, rad, dst, rrng, crng, rrng0, crng0)
    def _weights(self, x0, rad, rrng, crng):
        rad = rad.m # this should be in pixels when passed in
        cnst = -0.5 / (rad*rad)
        (xmsh,ymsh) = np.meshgrid(np.asarray(range(crng[0], crng[1]), dtype=np.float) - x0[1],
                                  np.asarray(range(rrng[0], rrng[1]), dtype=np.float) - x0[0])
        wmtx = np.exp(cnst * (xmsh**2 + ymsh**2))
        # We can trim off the extras now...
        min_w = np.exp(-0.5 * self.n_radii * self.n_radii)
        wmtx[wmtx < min_w] = 0.0
        wtot = wmtx.sum()
        if np.isclose(wtot, 0):
            wmtx *= 0
        else:
            wmtx /= wtot
        return wmtx
    def matrix(self, imshape, d2p):
        '''
        prf.matrix(im, d2p) or prf.matrix(im.shape, d2p) both yield a sparse matrix containing the
        weights for the given prf over the given image im (or an image of size imshape); the d2p
        parameter specifies the number of degrees per pixel for the image.
        '''
        if isinstance(imshape, np.ndarray): return self.matrix(imshape.shape)
        if len(imshape) > 2:                imshape = imshape[-2:]
        (x0, rad, dst, rrng, crng, _, _) = self._params(imshape, d2p)
        mini_mtx = self._weights(x0, rad, rrng, crng)
        mtx = sparse.lil_matrix(imshape)
        mtx[rrng[0]:rrng[1], crng[0]:crng[1]] = mini_mtx
        return mtx.asformat('csr')
    def __call__(self, im, d2p, c=None, edge_value=0, weights=True):
        '''
        prf(im, d2p) yields a tuple (u, w) in which u and w are equal-length vectors; the vector u
          contains the values found in the pRF while the equivalent vector w contains the matching
          weights for the values in u. The parameter d2p specifies the number of pixels per degree.
        prf(im, d2p, c) yields the weighted second moment about the value c*mu where mu is the
          weighted mean of the values in the pRF over the given image im.
        prf(im, d2p) is equivalent to prf(im, d2p, c=None) or prf(im, d2p, None).

        Note that the parameter im may be either a single image or an image stack (in which case it
        must be size (l, m, n) where l is the number of images and m and n are the rows and columns.
        Additionally, there is an optional parameter edge_value which is, by default, 0 and is
        used outside of the image range. Note that, because this function is generally meant to be
        used with contrast images, 0 is appropriate for the edge_value; it indicates that there is
        no contrast beyond the edge of the stimulus.

        The option weights may be set to false, in which case the weight parameter is not calculated
        and (u, None) is returned.
        '''
        # first we just grab out the values and weights; to do this we first grab the parameters:
        (x0, rad, dst, rrng, crng, rrng0, crng0) = self._params(im.shape, d2p)
        ## values are tricky because they may extend off the end:
        stackq = True if len(im.shape) == 3 else False
        u_im = im[:, rrng[0]:rrng[1], crng[0]:crng[1]] if stackq else \
               im[rrng[0]:rrng[1], crng[0]:crng[1]]
        if rrng[0] == rrng0[0] and rrng[1] == rrng0[1] and \
           crng[0] == crng0[0] and crng[1] == crng0[1]:
            u = u_im
        else:
            rows = rrng[1] - rrng[0]
            cols0 = crng0[1] - crng0[0]
            ev = edge_value
            # we may have extra bits on the top...
            u_r0 = np.full((rrng[0] - rrng0[0], cols0), ev, dtype=np.float) if rrng0[0] < rrng[0] \
                   else None
            # or on the bottom
            u_rr = np.full((rrng0[1] - rrng[1], cols0), ev, dtype=np.float) if rrng0[1] > rrng[1] \
                   else None
            # we might also have extra bits on the sides
            u_c0 = np.full((rows, crng[0] - crng0[0]), ev, dtype=np.float) if crng0[0] < crng[0] \
                   else [[] for r in range(rows)]
            u_cc = np.full((rows, crng0[1] - crng[1]), ev, dtype=np.float) if crng0[1] > crng[1] \
                   else [[] for r in range(rows)]
            # now put them together:
            if stackq:
                u_mid = np.asarray([np.concatenate((u_c0, uu, u_cc), axis=1) for uu in u_im])
                u = np.asarray(
                    [uu if len(row_cat) == 1 else np.concatenate(row_cat, axis=0)
                     for uu      in u_mid
                     for row_cat in [tuple([r for r in [u_r0, uu, u_rr] if r is not None])]])
            else:
                u_mid = np.concatenate((u_c0, u_im, u_cc), axis=1)
                row_cat = tuple([r for r in [u_r0, u_mid, u_rr] if r is not None])
                u = u_mid if len(row_cat) == 1 else np.concatenate(row_cat, axis=0)
        # Okay, now u is the correct size and w is the correct size...
        u = np.asarray([uu.flatten() for uu in u]) if stackq else u.flatten()
        if c is None and not weights: return (u, None)
        ## weights are from the _weights method:
        w = self._weights(x0, rad, rrng0, crng0)
        w = w.flatten()
        if c is None:
            return (u, w)
        elif stackq:
            return np.asarray([np.dot(w, (uu - c*np.dot(w, uu))**2) for uu in u])
        else:
            return np.dot(w, (u - c*np.dot(w, u))**2)

@pimms.calc('compressive_constants', cache=True)
def calc_compressive_constants(labels, compressive_constants_by_label):
    '''
    calc_compressive_constants is a calculator that combines the labels with the data in
    compressive_constants_by_label to produce a list of compressive constants.

    Required afferent parameters:
      * labels
      * compressive_constants_by_label

    Provided efferent values:
      @ compressive_constants Will be an array of values for the compressive constant n as used
        in the SOC model to compress the output (r = r0^n).
    '''
    n = np.asarray(lookup_labels(labels, compressive_constants_by_label), dtype=np.float)
    n.setflags(write=False)
    return n

@pimms.calc('pRF_sigmas', 'pRF_sigma_slopes', 'pRF_sigma_offsets', memoize=True, cache=True)
def calc_pRF_sigmas(labels, eccentricities, pRF_sigma_slopes_by_label, pRF_sigma_offsets_by_label):
    '''
    calc_pRF_sigmas is a calculator that combines the labels with the data in
    pRF_sigma_slopes_by_label to pRF_sigma_slopes, one slope per sigma; it also uses the
    eccentricity to determine the predicted sigma as well in pRF_sigmas.

    This calculation uses a simple formula:
      pRF_sigma = b[label] + m[label] * eccentricity
    where m[label] is the slope of the pRF size (vs eccentricity) in terms of the visual area label.
    The slopes, m, are given in the parameter pRF_sigma_slope_by_label.

    Required afferent parameters:
      * labels
      * pRF_sigma_slopes_by_label
      * eccentricities

    Provided efferent values:
      @ pRF_sigma_slopes Will be an array of values, one per pRF, of the sigma slope parameter;
        these parameters are used to predict pRF sigma values. (This value is m in the formula
        sigma = m * eccentricity + b.)
      @ pRF_sigma_offsets Will be an array of values, one per pRF, of the sigma offset parameter;
        these parameters are used to predict pRF sigma values. (This value is b in the formula
        sigma = m * eccentricity + b.)
      @ pRF_sigmas Will be set to an array of values, one per pRF, of the sigma parameter of each.
    '''
    ms = np.asarray(lookup_labels(labels, pRF_sigma_slopes_by_label),  dtype=np.float)
    bs = np.asarray(lookup_labels(labels, pRF_sigma_offsets_by_label), dtype=np.float)
    sigs = bs*units.degree + eccentricities*ms
    sigs.setflags(write=False)
    ms.setflags(write=False)
    bs.setflags(write=False)
    return (sigs, ms, bs)
        
@pimms.calc('pRF_centers', memoize=True, cache=True)
def calc_pRF_centers(polar_angles, eccentricities):
    '''
    calc_pRF_centers is a calculation that transforms polar_angles and eccentricities into
    pRF_centers, which are (x,y) coordinates, in degrees, in the visual field. 

    Required afferent parameters:
      @ polar_angles The polar angle values (obtained from import_benson14_from_freesurfer).
      @ eccentricities The eccentricity values (also from import_benson14_from_freesurfer).

    Efferent output values:
      @ pRF_centers Will be an n x 2 numpy matrix of the (x,y) pRF centers for each pRF in the
        visual field

    Notes:
      * The polar_angles and eccentricities parameters are expected to use pint's unit system and
        be either in degrees or radians
    '''
    if not pimms.is_quantity(polar_angles):
        warnings.warn('polar_angles is not a quantity; assuming that it is in degrees')
        polar_angles = polar_angles * units.degree
    if not pimms.is_quantity(eccentricities):
        warnings.warn('polar_angle is not a quantity; assuming that it is in degrees')
        eccentricities = eccentricities * units.degree
    angle_unit = eccentricities.u
    # Get the pRF centers:
    ang = np.pi/2 - polar_angles.to('rad').m
    xs = eccentricities * np.cos(ang)
    ys = eccentricities * np.sin(ang)
    pRF_centers = np.asarray([xs.to(angle_unit).m, ys.to(angle_unit).m]).T * angle_unit
    # That's it:
    return pRF_centers.to('deg')

@pimms.calc('pRFs', 'pRF_radii', memoize=True, cache=True)
def calc_pRFs(pRF_centers, pRF_sigmas, compressive_constants, pRF_n_radii=3.0):
    '''
    calc_pRFs is a calculator that adds to the datapool a set of objects of class PRFSpec;
    these objects represent the pRF and can calculate responses or sparse matrices representing
    the weights over an image of the pRF. Generally, if p is a PRFSpec object, then p(im, c) will
    yield the pRF response calculated in Kay et al (2013) with the parameter c; that is, it
    calculates the weighted second moment about c times the weighted mean of the pRF.

    Required afferent parameters:
      * pRF_centers, pRF_sigmas, compressive_constants

    Optional afferent parameters:
      @ pRF_n_radii May specify how many standard deviations should be included in the
        Gaussian blob that defines the pRF.

    Output efferent values:
      @ pRFs Will be the array of PRFSpec objects; this is a numpy array of the pRFs.
      @ pRF_radii Will be the effective pRF sizes, as determined by: radius = sigma / sqrt(n).
    '''
    prfs = np.asarray(
        [PRFSpec(x0, sig, n, n_radii=pRF_n_radii)
         for (x0, sig, n) in zip(pRF_centers, pRF_sigmas, compressive_constants)])
    prfs.setflags(write=False)
    radii = np.asarray([p.radius.to(units.deg).m for p in prfs]) * units.deg
    radii.setflags(write=False)
    return (prfs, radii)

@pimms.calc('cpd_sensitivities', cache=True)
def calc_cpd_sensitivities(labels, eccentricities, pRF_radii, cpd_sensitivity_function):
    '''
    calc_cpd_sensitivities is a calculator that produces the frequency sensitivity maps for each
    pRF.

    Required afferent parameters:
      * labels
      * eccentricities
      * pRF_radii
      * cpd_sensitivity_function

    Output efferent values:
      @ cpd_sensitivities Will be a persistent numpy array of maps, one per pRF, whose keys are
        frequencies (in cycles/degree) and whose values are the weights of that frequency for the
        relevant pRF.
    '''
    ss = np.full(len(labels), None, dtype=np.object)
    if isinstance(cpd_sensitivity_function, basestring):
        cpd_sensitivity_function = global_lookup(cpd_sensitivity_function)
    for (i, rad, ecc, lab) in zip(range(len(ss)), pRF_radii, eccentricities, labels):
        ss[i] = pyr.pmap(cpd_sensitivity_function(ecc, rad, lab))
    ss.setflags(write=False)
    return ss

