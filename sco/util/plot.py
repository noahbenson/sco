####################################################################################################
# sco/util/plot.py
# Plotting functions used in the SCO project
# by Noah C. Benson

import numpy             as np
import neuropythy        as ny
import pyrsistent        as pyr
import pint, pimms, os

def cortical_image(prediction, labels, pRFs, max_eccentricity, image_number=None,
                   visual_area=1, image_size=200, n_stds=1,
                   size_gain=1, method='triangulation', axes=None,
                   smoothing=None, cmap='afmhot', clipping=None, speckle=None):
    '''
    cortical_image(pred, labels, pRFs, maxecc) yields an array of figures that reconstruct the
      cortical images for the given sco results datapool. The image is always sized such that the
      width and height span 2 * max_eccentricity degrees (where max_eccentricity is stored in the
      datapool).

    The following options may be given:
      * visual_area (default: 1) must be 1, 2, or 3 and specifies whether to construct the cortical
        image according to V1, V2, or V3.
      * image_number (default: None) specifies which image to construct a cortical image of. If this
        is given as None, then a list of all images in the datapool is returned; otherwise only the
        figure for the image number specified is returned.
      * image_size (default: 200) specifies the width and height of the image in pixels (only for
        method equal to 'pRF_projection'.
      * n_stds (default: 1) specifies how many standard deviations should be used with each pRF when
        projecting the values from the cortical surface back into the image.
      * size_gain (default: 1) specifies a number to multiply the pRF size by when projecting into
        the image.
      * method (default: 'triangulation') should be either 'triangulation' or 'pRF_projection'. The
        former makes the plot by constructing the triangulation of the pRF centers and filling the
        triangles in while the latter creates an image matrix and projects the pRF predicted
        responses into the relevant pRFs.
      * cmap (default: matplotlib.cm.afmhot) specifies the colormap to use.
      * smoothing (default: None) if a number between 0 and 1, smoothes the data using the basic
        mesh smoothing routine in neurpythy with the given number as the smoothing ratio.
      * clipping (defaut: None) indicates the z-range of the data that should be plotted; this may
        be specified as None (don't clip the data) a tuple (minp, maxp) of the minimum and maximum
        percentile that should be used, or a list [min, max] of the min and max value that should be
        plotted.
      * speckle (default: None), if not None, must be an integer that gives the number points to
        randomly add before smoothing; these points essentially fill in space on the image if the
        vertices for a triangulation are sparse. This is only used if smoothing is also used.
    '''
    import matplotlib
    import matplotlib.pyplot as plt
    import matplotlib.tri    as tri
    import matplotlib.cm     as cm
    # if not given an image number, we want to iterate through all images:
    if image_number is None:
        return np.asarray([cortical_image(datapool, visual_area=visual_area, image_number=ii,
                                          image_size=image_size, n_stds=n_stds, method=method,
                                          axes=axes)
                           for ii in range(len(datapool['image_array']))])
    if axes is None: axes = plt.gca()
    # Some parameter handling:
    maxecc  = float(pimms.mag(max_eccentricity, 'deg'))
    centers = np.asarray([pimms.mag(p.center, 'deg') if l == visual_area else (0,0)
                          for (p,l) in zip(pRFs, labels)])
    sigs    = np.asarray([pimms.mag(p.radius, 'deg') if l == visual_area else 0
                          for (p,l) in zip(pRFs, labels)])
    z       = prediction[:,image_number]
    (x,y,z,sigs) = np.transpose([(xx,yy,zz,ss)
                                 for ((xx,yy),zz,ss,l) in zip(centers,z,sigs,labels)
                                 if l == visual_area and not np.isnan(zz)])
    clipfn = (lambda zz: (None,None))                if clipping is None            else \
             (lambda zz: np.percentile(z, clipping)) if isinstance(clipping, tuple) else \
             (lambda zz: clipping)
    cmap = getattr(cm, cmap) if pimms.is_str(cmap) else cmap
    if method == 'triangulation':
        if smoothing is None: t = tri.Triangulation(x,y)
        else:
            if speckle is None: t = tri.Triangulation(x,y)
            else:
                n0 = len(x)
                maxrad = np.sqrt(np.max(x**2 + y**2))
                (rr, rt) = (maxrad * np.random.random(speckle), np.pi*2*np.random.random(speckle))
                (x,y) = np.concatenate(([x,y], [rr*np.cos(rt), rr*np.sin(rt)]), axis=1)
                z = np.concatenate((z, np.full(speckle, np.inf)))
                t = tri.Triangulation(x,y)
            # make a cortical mesh
            coords = np.asarray([x,y])
            msh = ny.Mesh(coords, t.triangles.T)
            z = ny.mesh_smooth(msh, z, smoothness=smoothing, )
        (mn,mx) = clipfn(z)
        axes.tripcolor(t, z, cmap=cmap, shading='gouraud', vmin=mn, vmax=mx)
        axes.axis('equal')
        axes.axis('off')
        return plt.gcf()
    elif method == 'pRF_projection':
        # otherwise, we operate on a single image:    
        img        = np.zeros((image_size, image_size, 2))
        img_center = (float(image_size)*0.5, float(image_size)*0.5)
        img_scale  = (img_center[0]/maxecc, img_center[1]/maxecc)
        for (xx,yy,zz,ss) in zip(x,y,z,sigs):
            ss = ss * img_scale[0] * size_gain
            exp_const = -0.5/(ss * ss)
            row = yy*img_scale[0] + img_center[0]
            col = xx*img_scale[1] + img_center[1]
            if row < 0 or col < 0 or row >= image_size or col >= image_size: continue
            r0 = max([0,          int(round(row - ss))])
            rr = min([image_size, int(round(row + ss))])
            c0 = max([0,          int(round(col - ss))])
            cc = min([image_size, int(round(col + ss))])
            (mesh_xs, mesh_ys) = np.meshgrid(np.asarray(range(c0,cc), dtype=np.float) - col,
                                             np.asarray(range(r0,rr), dtype=np.float) - row)
            gaus = np.exp(exp_const * (mesh_xs**2 + mesh_ys**2))
            img[r0:rr, c0:cc, 0] += zz
            img[r0:rr, c0:cc, 1] += gaus
        img = np.flipud(img[:,:,0] / (img[:,:,1] + (1.0 - img[:,:,1].astype(bool))))
        fig = plotter.figure()
        (mn,mx) = clipfn(img.flatten())
        plotter.imshow(img, cmap=cmap, vmin=mn, vmax=mx)
        plotter.axis('equal')
        plotter.axis('off')
        return fig
    else:
        raise ValueError('unrecognized method: %s' % method)
    
def corrcoef_image(pred, meas, labels, pRFs, max_eccentricity,
                   visual_area=1, image_size=200, n_stds=1,
                   size_gain=1, method='triangulation', axes=None):
    '''
    corrcoef_image(imap) plots an image (using sco.util.cortical_image) of the correlation
      coefficients of the prediction versus the measurements in the given results map from
      the SCO model plan, imap.
    '''
    import matplotlib.pyplot as plt

    r = np.zeros(meas.shape[0])
    for (i,(p,t)) in enumerate(zip(pred,meas)):
            try:    r[i] = np.corrcoef(p,t)[0,1]
            except: r[0] = 0.0
    r = np.asarray([r]).T
    f = cortical_image(r, labels, pRFs, max_eccentricity, image_number=0,
                       visual_area=visual_area, image_size=image_size,
                       n_stds=n_stds, size_gain=size_gain, method=method,
                       axes=axes)
    axes = plt.gca() if axes is None else axes
    axes.clim((-1,1))
    return f

def report_image(anal):
    '''
    report_image(panam) createa a report image from the prediction analysis panal and returns the
      created figure.
    '''
    import matplotlib.pyplot as plt
    
    def analq(**kwargs): return anal[pyr.m(**kwargs)]
    
    eccs = np.unique([kk['eccentricity'] for kk in anal.iterkeys() if 'eccentricity' in kk])
    angs = np.unique([kk['polar_angle'] for kk in anal.iterkeys() if 'polar_angle' in kk])
    n_ec = len(eccs)
    n_pa = len(angs)

    (f,axs) = plt.subplots(4,2, figsize=(12,16))
    for (va,ax) in zip([1,2,3,None],axs):
        if va is None:
            r_ec = np.asarray([analq(eccentricity=e) for e in eccs])
            r_pa = np.asarray([analq(polar_angle=a) for a in angs])
        else:
            r_ec = np.asarray([analq(eccentricity=e, label=va) for e in eccs])
            r_pa = np.asarray([analq(polar_angle=a, label=va) for a in angs])
            e_pa = np.sqrt((1.0 - r_pa**2)/(n_pa - 2.0))
            e_ec = np.sqrt((1.0 - r_ec**2)/(n_ec - 2.0))

        ax[0].errorbar(eccs, r_ec, yerr=e_ec, c='r')
        if va is None: ax[0].set_xlabel('(All Areas) Eccentricity [deg]')
        else:          ax[0].set_xlabel('(V%d) Eccentricity [deg]' % va)
        ax[0].set_ylabel('Pearson\'s r')
        ax[0].set_ylim((-1.0,1.0))
        
        ax[1].errorbar(angs, r_pa, yerr=e_pa, c='r')
        if va is None: ax[1].set_xlabel('(All Areas) Polar Angle [deg]')
        else:          ax[1].set_xlabel('(V%d) Polar Angle [deg]' % va)
        ax[1].set_ylabel('Pearson\'s r')
        ax[1].set_xlim((-180.0,180.0))
        ax[1].set_ylim((-1.0,1.0))

    return f
