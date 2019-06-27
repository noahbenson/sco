####################################################################################################
# sco/util/plot.py
# Plotting functions used in the SCO project
# by Noah C. Benson

import numpy             as np
import neuropythy        as ny
import pyrsistent        as pyr
import pint, pimms, os, six

import matplotlib, matplotlib.colors, matplotlib.tri, matplotlib.collections
import matplotlib.pyplot as plt

blend_cmap = matplotlib.colors.LinearSegmentedColormap.from_list
cmap_sco = blend_cmap('sco', [(0,0,0), (0.5,0,0.5), (1,0,0), (1,1,0), (1,1,1)])

def vfield_plot(data, z='prediction', visual_area=1, image=0,
                labels='labels', coords='pRF_centers', max_eccentricity='max_eccentricity',
                smoothing=None, speckle=None, linewidth=1.0, linecolor='g',
                axes=None, cmap='sco', vmin=(2.5,), vmax=(97.5,)):
    '''
    vfield_plot(data) plots the given image data in the visual field on the current pyplot axes.

    The data argument may be any of the following:
      * An image matrix.
      * A 3xN or Nx3 matrix of [x,y,z] values where the x and y are visual field positions and the z
        is the response or height at the associated (x,y) position.
      * A map of {'x':..., 'y':..., 'z':...} providing the same information as the matrix; the 'z'
        key may be any of 'z', 'response', 'output', 'prediction', or any of these strings with an
        appended 's'.
      * A map of {'polar_angle':..., 'eccentricity':..., 'z':...} giving the polar angle and the
        eccentricity value of each of the visual field positions. The polar angle must be in degrees
        of rotation clockwise about the visual field where 0 degrees is the upper vertical meridian.
    
    The following optional arguments may be given:
      * z (default: 'prediction') specifies the data to use as the z-value of the image; this may be
        the key for an entry into the data map or an array of values; if data is an image, this is
        ignored.
      * visual_area (default: 1) specifies that only the given visual area should be plotted; this
        is ignored unless the data input was a map with the key 'labels' giving the visual_area
        labels per data-point.
      * image (default: 0) specifies the image to plot if the given z array contains multiple
        images.
      * labels (default: 'labels') specifies the data to use as the visual area labels in 
        conjunction with the visual_area argument; this may be the key for an entry into the data
        map or an array of values; if data is an image, this is ignored.
      * coords (default: 'pRF_centers') specifies the data to use as the coordinate points (pRF
        centers of the z values); this may be the key for an entry into the data map or an array of
        values; if data is an image, this is ignored.
      * max_eccentricity (default: None) specifies the max eccentricity that should be used; this
        is only relevant if the data input is not an image matrix.
      * smoothing (default: None) if a number between 0 and 1, smoothes the data using the basic
        mesh smoothing routine in neurpythy with the given number as the smoothing ratio.
      * speckle (default: None), if not None, must be an integer that gives the number points to
        randomly add before smoothing; these points essentially fill in space on the image if the
        vertices for a triangulation are sparse. This is only used if smoothing is also used. Note
        that if speckle is not None and smoothing is not None, then 90 points will be speckled
        around the max-eccentricity circle regardless of the speckle number; this is true even if
        speckle is 0 or False.
      * axes (default: None) specifies the set of pyplot axes on which to draw the image; if None
        uses pyplot.gca().
      * cmap (default: 'sco') specifies the colormap to use. The default SCO colormap is a
        dark-to-light colormap.
      * vmin and vmax (defaults: (2.5,) and (97.5,)) specify the min and max plot value. If given
        as a list or tuple with one entry, uses this value as a percentile limit cutoff.
    '''
    # Parse arguments
    if axes is None: axes = plt.gca()
    if cmap is None: cmap = cmap_sco
    elif pimms.is_str(cmap): cmap = cmap_sco if cmap.lower() == 'sco' else plt.get_cmap(cmap)
    mnmxfn = lambda z: [(f(z)                    if z0 is None          else
                         np.percentile(z, z0[0]) if pimms.is_vector(z0) else z0)
                        for (z0,f) in zip([vmin,vmax], [np.min, np.max])]
    def fromdata(name, k, defaults=[], unit=None):
        if k is None: return None
        elif pimms.is_vector(k): k = np.asarray(k)
        elif pimms.is_str(k):    k = data[k]
        elif k is Ellipsis:      k = next((data[kk] for k in dflts if kk in data), None)
        else: raise ValueError('Invalid %s value: %s' % (name, k))
        return None if k is None else pimms.mag(k) if unit is None else pimms.mag(k, unit)
    # okay, data might be an image or a matrix of [x, y, response] or a map of 'x','y', etc.
    if pimms.is_map(data):
        # Parse the arguments that only make sense in the context of an itable/map passed in:
        max_eccentricity = fromdata('max_eccentricity', max_eccentricity,
                                    ['maxecc','max_eccen','max_eccentricity'], 'deg')
        if max_eccentricity is not None: max_eccentricity = pimms.mag(max_eccentricity, 'deg')
        z = fromdata('response', z,
                     [k+sx for k in ['z','response','output','prediction'] for sx in ['','s']])
        z = np.asarray(z)
        (vmin,vmax) = mnmxfn(z)
        # few possibilities for x/y: coords gives us a matrix, or it's a tuple
        if coords is Ellipsis: coords = 'pRF_centers'
        if pimms.is_matrix(coords):        coords = pimms.mag(coords, 'deg')
        elif pimms.is_str(coords):         coords = pimms.mag(data[coords], 'deg')
        elif pimms.is_vector(coords, str) and len(coords) == 2:
            if (coords[0].startswith('ang') or '_ang' in coords[0] and
                coords[1].startswith('ecc') or '_ecc' in coords[1]):
                (ang,ecc) = [pimms.mag(x, 'deg') for x in coords]
                ang = np.pi/180.0 * (np.mod(90 - ang + 180, 360) - 180)
                coords = np.asarray([ecc*np.cos(ang), ecc*np.sin(ang)])
            elif (coords[1].startswith('ang') or '_ang' in coords[1] and
                  coords[0].startswith('ecc') or '_ecc' in coords[0]):
                (ecc,ang) = [pimms.mag(x, 'deg') for x in coords]
                ang = np.pi/180.0 * (np.mod(90 - ang + 180, 360) - 180)
                coords = np.asarray([ecc*np.cos(ang), ecc*np.sin(ang)])
            else:
                coords = [pimms.mag(data[c], 'deg') for c in coords]
        coords = np.asarray(coords)
        if coords.shape[0] != 2: coords = coords.T
        # the z might have been a list of values per image, in which case we need to interpret the
        # image number:
        if len(z.shape) == 2:
            if z.shape[1] != coords.shape[1]: z = z.T
            z = z[image]
        # get the labels as well
        lbl = fromdata('labels', labels, ['labels', 'lbls', 'label', 'lbl'])
        # organize into a matrix
        data = np.vstack([coords, z])
        if visual_area is not None and lbl is not None:
            if pimms.is_number(visual_area):   data = data[:, lbl == visual_area]
            elif pimms.is_vector(visual_area): data = data[:, np.isin(lbl, visual_area)]
            else: raise ValueError('Cannot interpret visual_area: %s' % visual_area)
    # now we've processed map inputs; how about 
    if pimms.is_matrix(data):
        data = np.asarray(data)
        if data.shape[0] == 3 or data.shape[1] == 3:
            data = data if data.shape[0] == 3 else data.T
            if max_eccentricity is not None:
                ecc = np.sqrt(np.sum(data[:2]**2, axis=0))
                data = data[:, ecc <= max_eccentricity]
            else: max_eccentricity = np.max(np.sqrt(np.sum(data[:2]**2, 0)))
            (x,y,z) = data
            (x,y) = [pimms.mag(u,'deg') for u in (x,y)]
            # we need to make a triangulation on the axis; if smoothing and speckle were given, we
            # need to handle those as well
            if smoothing is not None:
                if speckle is not None:
                    n0 = len(x)
                    maxrad = (np.sqrt(np.max(x**2 + y**2)) if max_eccentricity is None else
                              max_eccentricity)
                    (rr,rt) = (maxrad*np.random.random(speckle), np.pi*2*np.random.random(speckle))
                    # add points around the boundary
                    rr = np.concatenate([rr, np.full(90, max_eccentricity)])
                    rt = np.concatenate([rt, np.linspace(0, 2*np.pi, 91)[:-1]])
                    (x,y) = np.concatenate(([x,y], [rr*np.cos(rt), rr*np.sin(rt)]), axis=1)
                    z = np.concatenate((z, np.full(len(rr), np.inf)))
                t = matplotlib.tri.Triangulation(x,y)
                # make a cortical mesh and smooth it
                coords = np.asarray([x,y])
                msh = ny.mesh(t.triangles.T, coords)
                z = msh.smooth(z, smoothness=smoothing)
            tri = matplotlib.tri.Triangulation(x, y)
            # before we plot anything, put down a black (or null-colored) circle
            rr  = np.max(np.sqrt(np.sum(coords**2, 0)))
            clr0 = cmap(0)
            bg = axes.add_patch(plt.Circle([0,0], rr, edgecolor=None, facecolor=clr0, fill=True))
            bg.set_zorder(0)
            fg = axes.tripcolor(tri, z, shading='gouraud', cmap=cmap, vmin=vmin, vmax=vmax)
            fg.set_zorder(1)
            # last thing: if the linecolor and width aren't None, plot lines:
            if linewidth is not None and linecolor is not None:
                axes.plot([0,0], [-rr,rr], '-', c=linecolor, lw=linewidth)
                if visual_area is not None and visual_area != 1:
                    axes.plot([-rr,rr], [0,0], '-', c=linecolor, lw=linewidth)
                l = plt.Circle([0,0],rr, edgecolor=linecolor, fill=False, lw=linewidth)
                l = axes.add_patch(l)
                l.set_zorder(2)
        else:
            # we need to make an image
            axes.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
    else: raise ValueError('Cannot interpret input image data')

def cortex_plot(data, hemi, z='prediction', image=0,
                cortex_indices='cortex_indices', labels='labels', hemispheres='hemispheres',
                subject='freesurfer_subject', max_eccentricity='max_eccentricity',
                eccentricities='eccentricities', axes=None, cmap='sco', vmin=(2.5,), vmax=(97.5,),
                linewidth=0.5, linecolor='g'):
    '''
    cortex_plot(data, hemi) plots the given image data on the associated cortex for the given hemi.

    The data argument must be a map or mapping with keys containing the relevant data, as indicated
    by the various options.
    
    The following optional arguments may be given:
      * z (default: 'prediction') specifies the data to use as the z-value of the image; this may be
        the key for an entry into the data map or an array of values; if data is an image, this is
        ignored.
      * image (default: 0) specifies the image to plot if the given z array contains multiple
        images.
      * labels (default: 'labels') specifies the data to use as the visual area labels in 
        conjunction with the visual_area argument; this may be the key for an entry into the data
        map or an array of values; if data is an image, this is ignored.
      * cortex_indices (default: 'cortex_indices') specifies the surface vertices that correspond to
        the given z values.
      * hemispheres (defaut: 'hemispheres') specifies the hemispheres (1 for LH, -1 for RH) that
        should be used.
      * subject (default: 'freesurfer_subject') specifies the subject that should be used; this is
        assumed to be an entry into data if it is a string; however, if it is a string and it is not
        one of the keys of data, then it is tried as a FreeSurfer subject ID or path.
      * max_eccentricity (default: None) specifies the max eccentricity that should be used; this
        is only relevant if the data input is not an image matrix.
      * axes (default: None) specifies the set of pyplot axes on which to draw the image; if None
        uses pyplot.gca().
      * cmap (default: 'sco') specifies the colormap to use. The default SCO colormap is a
        dark-to-light colormap.
      * vmin and vmax (defaults: (2.5,) and (97.5,)) specify the min and max plot value. If given
        as a list or tuple with one entry, uses this value as a percentile limit cutoff.
    '''
    # sometimes we get two axes and a none or lr for hemi:
    if hemi is None or (pimms.is_str(hemi) and hemi.lower() == 'lr'):
        if not pimms.is_vector(axes): axes = (axes,axes)
        return tuple([cortex_plot(data, h, z=z, image=image, axes=ax,
                                  cortex_indices=cortex_indices, labels=labels,
                                  hemispheres=hemispheres, subject=subject,
                                  max_eccentricity=max_eccentricity,
                                  cmap=cmap, vmin=vmin, vmax=vmax)
                      for (h,ax) in zip(['lh','rh'], axes)])
    # Parse arguments
    if axes is None: axes = plt.gca()
    if cmap is None: cmap = cmap_sco
    elif pimms.is_str(cmap): cmap = cmap_sco if cmap.lower() == 'sco' else plt.get_cmap(cmap)
    mnmxfn = lambda z: [(f(z)                    if z0 is None          else
                         np.percentile(z, z0[0]) if pimms.is_vector(z0) else z0)
                        for (z0,f) in zip([vmin,vmax], [np.min, np.max])]
    def fromdata(name, k, defaults=[], unit=None):
        if k is None: return None
        elif pimms.is_vector(k): k = k
        elif pimms.is_str(k):    k = data[k]
        elif k is Ellipsis:      k = next((data[kk] for k in dflts if kk in data), None)
        else: raise ValueError('Invalid %s value: %s' % (name, k))
        return (None if k is None else np.asarray(pimms.mag(k)) if unit is None else
                np.asarray(pimms.mag(k, unit)))
    max_eccentricity = fromdata('max_eccentricity', max_eccentricity,
                                ['maxecc','max_eccen','max_eccentricity'], 'deg')
    if max_eccentricity is not None: max_eccentricity = pimms.mag(max_eccentricity, 'deg')
    # Okay, let's get the data arguments:
    z = fromdata('response', z, ['z','response','output','prediction'])
    z = np.asarray(z)
    cidcs = fromdata('cortex_indices', cortex_indices, ['cortex_indices'])
    hemis = fromdata('hemispheres', hemispheres, ['hemispheres', 'hemis'])
    lbls  = fromdata('labels', labels, ['labels', 'pRF_labels', 'visual_areas'])
    eccs  = fromdata('eccen', eccentricities, ['eccentricities', 'eccen', 'ecc'], 'deg')
    if len(cidcs.shape) == 2:
        raise ValueError('cortex_plot can only be used with surface-based modalities')
    # we know what we're plotting now, so let's get vmin and vmax
    (vmin, vmax) = mnmxfn(z)
    # the z might have been a list of values per image, in which case we need to interpret the
    # image number:
    if len(z.shape) == 2:
        if z.shape[0] != len(cidcs): z = z.T
        z = z[:,image]
    # Okay, trim by hemisphere
    h = hemi if hemi in (-1,1) else {'lh':1, 'rh':-1}[hemi.lower()]
    hemi = {1:'lh', -1:'rh'}[h]
    if pimms.is_str(hemis[0]):
        (h_old,hemis) = (hemis, np.ones(len(hemis), dtype='int'))
        hemis[np.isin(h_old, ['rh','RH','r','R'])] = -1
    ii = (hemis == h)
    # and by max eccentricity
    if max_eccentricity is not None and eccs is not None:
        ii &= (eccs <= max_eccentricity)
    ii = np.where(ii)[0]
    (cidcs,lbls,eccs,z) = [u[ii] for u in (cidcs,lbls,eccs,z)]
    # okay, next task is to deal with the subject and map projections, etc.
    # we start with the native sphere:
    if pimms.is_str(subject):
        if subject in data: subject = data[subject]
        if pimms.is_str(subject): sub = ny.freesurfer_subject(subject)
        else: sub = subject
    else: sub = subject
    sph = sub.hemis[hemi].registrations['native']
    xyz = sph.coordinates
    # make a map projection; we want the middle of activatable v1 in the center...
    x0 = np.mean(xyz[:, cidcs], axis=1)
    x0 /= np.sqrt(np.sum(x0**2))
    # and we want the peripheral tip of V1 on the positive x (for LH)
    highecc = np.percentile(eccs[lbls == 1], 95)
    x1 = np.mean(xyz[:, cidcs[(lbls == 1) & (eccs > highecc)]], axis=1)
    x1 /= np.sqrt(np.sum(x1**2))
    if hemi == 'rh': # point across for RH
        th = np.arccos(np.dot(x0, x1))
        x1 = np.dot(ny.geometry.rotation_matrix_3D(np.cross(x1,x0), 2*th), x1)
    # what's the minimum radius in radians that we need for the whole V1/2/3 region?
    pts = xyz[:,cidcs]
    pts = pts / np.sqrt(np.sum(pts**2, axis=0))
    ths = np.arccos(np.dot(pts.T, x0))
    th = 1.15 * np.max(ths) # this is what we use
    # make the map projection and the map
    mp = ny.map_projection(center=100*x0, center_right=100*x1,
                           method='orthographic', radius=th, chirality=hemi)
    m = mp(sph)
    xy = m.coordinates
    dxy = 0.025 * (np.max(xy[0]) - np.min(xy[0]))
    if hemi == 'lh': m = m.copy(coordinates=(xy + dxy - np.min(xy[0])))
    else:            m = m.copy(coordinates=(xy - dxy - np.max(xy[0])))
    # convert our z values and our labels into map-indexing...
    cidcs = m.tess.index(cidcs)
    (zs,ws,ls) = np.zeros([3, m.vertex_count])
    ws[cidcs] = 1
    ls[cidcs] = lbls
    # prepare the boundary plot...
    lc = []
    xy = m.coordinates
    fs = m.tess.indexed_faces
    for va in (1,2,3):
        fws = np.asarray([ls[f] == va for f in fs])
        fwstot = np.sum(fws, axis=0)
        for (inum,iflag) in [(1,0), (2,1)]:
            iitot = np.where(fwstot == inum)[0]
            for dd in (0,1,2):
                ii = iitot[fws[dd,iitot] == iflag]
                oth = np.setdiff1d([0,1,2], [dd])
                (a,(b,c)) = (xy[:,fs[dd,ii]], [xy[:,fs[oo,ii]] for oo in oth])
                lc.append(0.5 * np.transpose([a+b, a+c], (2,0,1)))
    lc = np.concatenate(lc, 0)
    lc = matplotlib.collections.LineCollection(lc, colors=linecolor, linewidths=linewidth)
    # here's how we plot...
    def plotfn(ax,z):
        zs[cidcs] = z
        pp = ny.cortex_plot(m, color=zs, alpha=ws, cmap=cmap, vmin=vmin, vmax=vmax, axes=ax)
        ax.add_collection(lc)
        return pp
    # and make a plot(s)...
    if pimms.is_vector(axes):
        if not pimms.is_matrix(z): z = np.asarray([z])
        (res,n) = ([], len(z))
        for (k,ax) in enumerate(axes):
            zs[cidcs] = z[k % n]
            res.append(plotfn(ax,zs))
    else: res = plotfn(axes, z)
    return res

def responses_plot(data, z='prediction', image=0, image_array='image_array',
                   cortex_indices='cortex_indices', labels='labels', hemispheres='hemispheres',
                   subject='freesurfer_subject', max_eccentricity='max_eccentricity',
                   eccentricities='eccentricities', coords='pRF_centers', axes=None, cmap='sco',
                   vmin=(2.5,), vmax=(97.5,), linewidth=1.0, linecolor='g',
                   smoothing=0.5, speckle=0):
    '''
    responses_plot(data) plots the given image data on a 2x3 matrix of axes; this proceeds by 
      calling cortex_plot() and vfield_plot() for each.

    The data argument must be a map or mapping with keys containing the relevant data, as indicated
    by the various options.
    
    The following optional arguments may be given:
      * z (default: 'prediction') specifies the data to use as the z-value of the image; this may be
        the key for an entry into the data map or an array of values; if data is an image, this is
        ignored.
      * image (default: 0) specifies the image to plot if the given z array contains multiple
        images.
      * labels (default: 'labels') specifies the data to use as the visual area labels in 
        conjunction with the visual_area argument; this may be the key for an entry into the data
        map or an array of values; if data is an image, this is ignored.
      * cortex_indices (default: 'cortex_indices') specifies the surface vertices that correspond to
        the given z values.
      * hemispheres (defaut: 'hemispheres') specifies the hemispheres (1 for LH, -1 for RH) that
        should be used.
      * subject (default: 'freesurfer_subject') specifies the subject that should be used; this is
        assumed to be an entry into data if it is a string; however, if it is a string and it is not
        one of the keys of data, then it is tried as a FreeSurfer subject ID or path.
      * max_eccentricity (default: None) specifies the max eccentricity that should be used; this
        is only relevant if the data input is not an image matrix.
      * axes (default: None) specifies the set of pyplot axes on which to draw the image; if None,
        fills the current figure with axes. The axes may be given as a list 2x3 matrix of axes or a
        list of 6 axes or as a tuple of (figure, rect) in which case rect is [left,bottom,w,h] and
        six axes are created within the given rect of the figure.
      * cmap (default: 'sco') specifies the colormap to use. The default SCO colormap is a
        dark-to-light colormap.
      * vmin and vmax (defaults: (2.5,) and (97.5,)) specify the min and max plot value. If given
        as a list or tuple with one entry, uses this value as a percentile limit cutoff.
    '''
    # process axes...
    if axes is None:
        (fig,dw,dh) = (plt.gcf(), 1.0/3.0, 0.5)
        axes = [[fig.add_axes([dw*col, 1 - dh*row, dw, dh]) for col in [0,1,2]] for row in [1,2]]
        axes = np.reshape(axes, (1,2,3))
    elif pimms.is_vector(axes):
        if len(axes) == 2:
            (fig,(l,b,w,h)) = axes
            (dw,dh) = (w/3.0, h/2.0)
            axes = [[fig.add_axes([l+dw*col, b + h - dh*row, dw, dh]) for col in [0,1,2]]
                    for row in [0,1]]
        elif len(axes) == 6: axes = [axes[:3],axes[3:]]
        else: raise ValueError('Six axes must be given')
        axes = np.asarray([axes])
    elif pimms.is_array(axes) and len(np.shape(axes)) == 3:
        axes = np.asarray(axes)
        if axes.shape[1:] != (2,3): raise ValueError('3D axes arg must be N x 2 x 3')
    elif pimms.is_matrix(axes): axes = np.asarray([axes])
    elif isinstance(axes, matplotlib.figure.Figure):
        (fig,dw,dh) = (axes, 1.0/3.0, 0.5)
        axes = [[fig.add_axes([dw*col, 1 - dh*row, dw, dh]) for col in [0,1,2]] for row in [1,2]]
        axes = np.reshape(axes, (1,2,3))
    else: raise ValueError('Axes must be a fig+rect or a set of 6 axes')
    # parse the image_array name:
    if pimms.is_str(image_array): image_array = data[image_array]
    axes = np.transpose(axes, (1,2,0))
    # if there are multiple images/axes, we do image array plot differently:
    image = np.reshape(image,(-1,))
    if axes.shape[2:] != image.shape: raise ValueError('images and axes count must match')
    for (ii,imno) in enumerate(image):
        ax = axes[0,0,ii]
        if ax is not None: vfield_plot(image_array[imno], cmap='gray', axes=ax, vmin=0, vmax=1)
        for va in (1,2,3):
            ax = axes[1,va-1,ii]
            if ax is None: continue
            vfield_plot(data, z, visual_area=va, image=imno, axes=ax,
                        labels=labels, coords=coords, max_eccentricity=max_eccentricity,
                        linewidth=linewidth, linecolor=linecolor, speckle=speckle,
                        smoothing=smoothing, cmap=cmap, vmin=vmin, vmax=vmax)
    if len(image) == 1: (image,axes) = [np.squeeze(x) for x in (image,axes)]
    for (h,q) in zip(['lh','rh'], [1,2]):
        ax = axes[0,q]
        if ax is None: continue
        cortex_plot(data, h, z=z, image=image, axes=ax,
                    cortex_indices=cortex_indices, labels=labels, hemispheres=hemispheres,
                    subject=subject, max_eccentricity=max_eccentricity,
                    eccentricities=eccentricities, linecolor=linecolor,
                    linewidth=linewidth, vmin=vmin, vmax=vmax, cmap=cmap)
    for ax in axes.flatten():
        if ax is not None: ax.axis('off');

def draw_gabor_schema(axes=None, theta=0, scale=1, center=(0.12,0.12), size=0.2):
    '''
    draw_gabor_schema(axes) draws a gabor schema in the upper-right corner of the given axes.
    draw_gabor_schema() uses the current axes.
    draw_gabor_schema(axes, th) uses the angle th in radians to represent an oriented gabor.
    draw_gabor_schema(axes, th, s) scales the gabor schema's ripple representation in size linearly
      in proportion with s; generally, s should be less than or equal to 1.
    '''
    # parse the axes arg:
    if axes is None: axes = plt.gca()
    # get axes plot range
    ((r0,r1),(c0,c1)) = [reversed(axes.get_ylim()), axes.get_xlim()]
    (rs,cs) = (r1-r0, c1-c0)
    # draw the overall circle
    (rc,cc) = (center[0]*rs + r0, center[1]*cs + c0)
    el = matplotlib.patches.Ellipse([cc,rc], size*cs, size*rs,
                                    fill=True, edgecolor='w', linewidth=0.5,
                                    facecolor=(0.5,0.5,0.5))
    axes.add_patch(el)
    # okay, figure out how to do the little ellipses
    dc = 0.2 * size * cs * scale
    # the negative th fixes the fact that the y-axis is reversed in these images
    th = -theta
    a = th * 180/np.pi
    rmtx = (lambda c,s: [[c,-s],[s,c]])(np.cos(th), np.sin(th))
    (cmn,rmn) = np.dot(rmtx, [-dc,0]) + [cc,rc]
    (cmx,rmx) = np.dot(rmtx, [dc, 0]) + [cc,rc]
    elfn = lambda c,r,kc,kr,clr: matplotlib.patches.Ellipse(
        [c,r], kc*cs, kr*rs, a, fill=True, edgecolor=None, facecolor=clr)
    (cws,rws) = 1.2 * np.array([[0.3, 0.25, 0.15], [0.7, 0.5, 0.3]])
    ss = size * scale
    for (cw,rw,clr) in zip(cws,rws,['0.333', '0.167', '0']):
        axes.add_patch(elfn(cmn,rmn, cw*ss, rw*ss, clr))
    for (cw,rw,clr) in zip(cws,rws,['0.667', '0.883', '1']):
        axes.add_patch(elfn(cmx,rmx, cw*ss, rw*ss, clr))
    return axes

def contrast_energies_plot(data, image=0, axes=None, spatial_frequencies=0.5,
                           orientations=Ellipsis,
                           gabor_spatial_frequencies='gabor_spatial_frequencies',
                           gabor_orientations='gabor_orientations',
                           oriented_contrast_images='oriented_contrast_images',
                           label_rows=True):
    '''
    contrast_energies_plot(data, imageno) plots a schema of contrast images for the given
      image along with gabor schemas.
    '''
    def fromdata(k):
        if k is None: return None
        if pimms.is_str(k): return data[k]
        else: return k
    from skimage.transform import pyramid_expand
    ors  = fromdata(gabor_orientations) if orientations is Ellipsis else orientations
    ors  = pimms.mag(ors, 'deg')
    sfs0 = pimms.mag(fromdata(gabor_spatial_frequencies), 'cycles/deg')
    sfs  = sfs0
    if   spatial_frequencies is Ellipsis: pass
    elif pimms.is_number(spatial_frequencies):
        (k, n) = (spatial_frequencies, len(sfs))
        if not pimms.is_int(k) or k < 1: k = int(np.round(len(ors) * k))
        ii = np.round(np.linspace(0, n-1, k*2+1)[1:-1:2]).astype('int')
        sfs = sfs[ii]
    elif pimms.is_vector(spatial_frequencies, 'number'):
        sfs = pimms.mag(spatial_frequencies, 'cycles/deg')
    else: raise ValueError('cannot interpret spatial_frequencies: %s' % spatial_frequencies)
    sfs = np.asarray(sfs)
    (rs,cs) = (len(sfs), len(ors))
    if axes is None or pimms.is_vector(axes) and len(axes) == 2:
        (fig,(l,b,w,h)) = (plt.gcf(), [0,0,1,1]) if axes is None else axes
        (ws,hs) = (0.01,0.01)
        (dr,dc) = ((h-hs)/rs, (w-ws)/cs)
        (r0,c0) = (hs+b, ws+l)
        axs = [[fig.add_axes([c0+th*dc, r0 + sf*dr, dc-ws, dr-hs])
                for th in range(cs)] for sf in range(rs)]
        axs = np.asarray(axs)
    elif pimms.is_array(axes): axes = np.reshape(axes, (rs,cs))
    ocims = fromdata(oriented_contrast_images)
    sfszs = np.flip(np.linspace(0.3, 1, rs))
    sftr = {k:v for (v,k) in enumerate(sfs0)}
    im0 = sorted(ocims, key=lambda x:-x.shape[-1])[0][0,0]
    for (sfii,sfsz,axrow) in zip(range(rs), sfszs, axs):
        for (orii, ax) in enumerate(axrow):
            im = np.abs(ocims[sftr[sfs[sfii]]][orii][image])
            if im.shape[0] < im0.shape[0]:
                im = np.array(im)
                im = pyramid_expand(im, im0.shape[0]/im.shape[0], multichannel=False)
            ax.imshow(im, cmap='gray', vmin=0, vmax=0.2).set_zorder(-10)
            ax.axis('off');
    # Annotate the cpd
    if label_rows:
        for (sf,ax) in zip(sfs,axs[:,0]):
            (y1,y0) = ax.get_ylim()
            (x0,x1) = ax.get_xlim()
            (xx,yy) = (x1 - x0, y1 - y0)
            ax.text(x0+0.05*xx, y1 - 0.025*yy, '%04.2f cpd' % sf,
                    color='g', size=8, va='bottom', ha='left')
    for (sfii,sfsz,axrow) in zip(range(rs), sfszs, axs):
        for (orii, ax) in enumerate(axrow):
            draw_gabor_schema(ax, theta=ors[orii]*np.pi/180, scale=sfsz)
    
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
    import warnings
    warnings.warn('cortical_image is deprecated; use sco.util.cortex_image instead',
                  DeprecationWarning)
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
            msh = ny.mesh(coords, t.triangles.T)
            z = msh.smooth(msh, z, smoothness=smoothing)
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
    
    eccs = np.unique([kk['eccentricity'] for kk in six.iterkeys(anal) if 'eccentricity' in kk])
    angs = np.unique([kk['polar_angle'] for kk in six.iterkeys(anal) if 'polar_angle' in kk])
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
