####################################################################################################
# __main__.py
# The main function, if sco is invoked directly as command.
# By Noah C. Benson

import os, sys, six, tempfile, tarfile

from neuropythy.util import CommandLineParser
from sco import (build_model)

main_parser = CommandLineParser(
    [('h', 'help',                           'help',                           False),
     ('C', 'create-directories',             'create_directories',             False),
     ('u', 'use-spatial-gabors',             'use_spatial_gabors',             False),
     
     ('w', 'aperture-edge-width',            'aperture_edge_width',            Ellipsis),
     ('r', 'aperture-radius',                'aperture_radius',                Ellipsis),
     ('b', 'background',                     'background',                     Ellipsis),
     ('n', 'compressive-constants-by-label', 'compressive_constants_by_label', Ellipsis),
     ('c', 'contrast-constants-by-label',    'contrast_constants_by_label',    Ellipsis),
     ('f', 'cpd-sensitivity-function',       'cpd_sensitivity_function',       Ellipsis),
     ('R', 'divisive-exponents-by-label',    'divisive_exponents_by_label',    Ellipsis),
     ('d', 'divisive-normalization-schema',  'divisive_normalization_schema',  Ellipsis),
     ('G', 'gabor-orientations',             'gabor_orientations',             Ellipsis),
     ('k', 'gains-by-label',                 'gains_by_label',                 Ellipsis),
     ('g', 'gamma',                          'gamma',                          Ellipsis),
     ('i', 'import-filter',                  'import_filter',                  Ellipsis),
     ('e', 'max-eccentricity',               'max_eccentricity',               Ellipsis),
     ('M', 'measurements-filename',          'measurements_filename',          Ellipsis),
     ('m', 'modality',                       'modality',                       Ellipsis),
     ('X', 'normalized-pixels-per-degree',   'normalized_pixels_per_degree',   Ellipsis),
     ('o', 'output-directory',               'output_directory',               Ellipsis),
     ('P', 'output-prefix',                  'output_prefix',                  Ellipsis),
     ('S', 'output-suffix',                  'output_suffix',                  Ellipsis),
     ('N', 'pRF-n-radii',                    'pRF_n_radii',                    Ellipsis),
     ('Q', 'pRF-sigma-offsets-by-label',     'pRF_sigma_offsets_by_label',     Ellipsis),
     ('q', 'pRF-sigma-slopes-by-label',      'pRF_sigma_slopes_by_label',      Ellipsis),
     ('s', 'saturation-constants-by-label',  'saturation_constants_by_label',  Ellipsis)])

sco_help = \
'''
Usage: python -m sco <subject> <pixels-per-degree> <image0000> <image0001>...
Runs the Standard Cortical Observer prediction routine and exports either a
prediction volume file (predictions.mgz) or a pair of prediction surface files
(lh.predictions.mgz and rh.predictions.mgz).

The following options may be given.
================================================================================
  * -h|--help
    Prints this message.
  * -C|--create-directories
    Indicates that the output directory should be auto-created if it does not
    exist; by default this results in an error being raised.
  * -u|--use-spatial-gabors
    If provided, uses spatial gabor filters instead of the steerable pyramid.
  * -w<x>|--aperture-edge-width=<x>
    Specifies that the aperture edge width should be <x> degrees. Application of
    an aperture edge consists of blending the edge of the stimulus aperture into
    the background in order to minimize edge effects of the stimulus. By default
    no aperture is used. Note that the aperture edge is applied to the <x> 
    degrees inside the aperture; it does not extend the aperture radius.
  * -r<x>|--aperture-radius=<x>
    Specifies that an aperture with radius <x> (in degrees) should be applied to
    the stimulus images; beyond <x> degrees, the stimulus image will consist of
    background. Note that the aperture_edge_width is applied to the inner part
    of the aperture and does not extend the aperture_radius.
  * -b<v>|--background=<v>
    Specifies the background gray-level value. By default this is 0.5 (gray).
  * -n<s>|--compressive-constatns-by-label=<s>
    Specifies the compressive constants (aka, the output nonlinearity) to be
    used. Compressive constants closer to 1 result in more linear additive
    responses while compressive constants closer to 0 result in more sub-
    additive responses. By default, this uses the values in 
    sco.impl.benson17.compressive_constants_by_label_Kay2013, which were derived
    from Kay et al. (2013). These must be provided in a 'by-label' format (see
    "by-label data" below).
  * -c<s>|--contrast-constants-by-label=<s>
    Specifies the contrast constants (aka the second-order contrast weight) by
    visual area. Values of c closer to 0 indicate heavier reliance on
    first-order contrast while values closer to 1 indicate heavier reliance on
    second-order contrast. By default, this uses the values in 
    sco.impl.benson17.contrast_constants_by_label_Kay2013, which were derived
    from Kay et al. (2013). These must be provided in a 'by-label' format (see
    "by-label data" below).
  * -f<s>|--cpd-sensitivity-function=<s>
    Specifies the function to use for determining the sensitivity to spatial
    frequency (in cycles-per-degree). This must be the name of a python function
    and by default is 'sco.impl.benson17.cpd_sensitivity.
  * -R<s>|--divisive-exponents-by-label=<s>
    Specifies the divisive normalization step's exponent in terms of visual area
    label. By default this uses the values in the variable 
    sco.impl.benson17.divisive_exponents_by_label_Kay2013, which were derived
    from Kay et al. (2013). These must be provided in a 'by-label' format (see
    "by-label data" below).
  * -d<s>|--divisive-normalization-schema=<s>
    Specifies the divisive normalization schema to use. By default, this uses
    the 'Heeger1992' schema. The value specified must be a known named schema,
    so valid values are 'Heeger1992' and 'naive'.
  * -G<n>|--gabor-orientations=<n>
    Specifies the number of gabor orientations to use. By default this is 8.
  * -k<s>|--gains-by-label=<s>
    Specifies the gain to use per visual area. By default this uses the values
    in the variable sco.impl.benson17.gains_by_label_Benson2017, which are all
    equal to 1. These must be provided in a 'by-label' format (see
    "by-label data" below).
  * -g<s>|--gamma=<s>
    Specifies the gamma correction. See "Specifying the Gamma Table" below.
  * -i<s>|--import-filter=<s>
    Specifies the import filter; this must be the name of a python variable that
    contains a function f(polar_angle, eccentricity, label, hemi); f must yield
    True for voxels/vertices that should be included and False for others.
  * -e<x>|--max-eccentricity=<x>
    Specifies that the maximum eccentricity value for which predictions should
    be calculated; voxels and vertices with values above this are ignored. By
    default this is 12.
  * -M<s>|--measurements-filename=<s>
    Specifies the measurement filename or filenames. If the modality used is
    surface, then this must contain two colon-separated files, one for the LH
    and one for the RH. These files must be in Freesurfer orientation (for
    volume) or must be organized as (1 x 1 x n x k) volumes for surface
    modality where n is the number of vertices in the appropriate surface and k
    is the number of images. Each value should specify the measured response of
    the appropriate voxel/vertex for the appropriate image. By default,
    no measurement files are used and no comparison data are exported.
  * -m<s>|--modality=<s>
    May specify the modality as 'surface' or 'volume'. The default is 'surface',
    which calculates predictions for all vertices while 'volume' calculates
    predictions for all ribbon voxels.
  * -X<x>|--normalized-pixels-per-degree=<x>
    Specifies the number of pixels per degree after image normalization; this
    option may be used to down-sample large images prior to processing. By
    default, this uses the same value as provided in the pixels-per-degree
    argument.
  * -o<s>|--output-directory=<s>
    Specifies the directory to which to write the output files. By default this
    is the current directory (.).
  * -P<s>|--output-prefix=<s>
    Specifies that output files should contain the given prefix <s>. For example
    if the option -Ptest_ is given for a surface modality, then the output files
    will be named lh.test_predictions.mgz and rh.test_predictions.mgz.
  * -S<s>|--output-suffix=<s>
    Specifies that output files should contain the given suffix <s>. For example
    if the option -P_test is given for a surface modality, then the output files
    will be named lh.predictions_test.mgz and rh.predictions_test.mgz.
  * -N<x>|--pRF-n-radii=<x>
    Specifies the number of pRF radii to include in calculations; by default
    this is 3. Larger values will include more low-weight pixels in the
    calculation of spatial frequencies, potentially increasing the accuracy but
    requiring more computation time. Generally it is not necessary to change
    this value.
  * -Q<s>|--pRF-sigma-offsets-by-label=<s>
  * -q<s>|--pRF-sigma-slopes-by-label=<s>
    Specifies the sigma (pRF radius) offset/slope by visual area. pRF sizes are
    calculated according to the formula m*e + b where m is the pRF-sigma-slope,
    b is the pRF-sigma-offset, and e is the eccentricity. By default, this uses
    the values in sco.impl.benson17.pRF_sigma_offsets_by_label_Wandell2015 and
    the values in sco.impl.benson17.pRF_sigma_slopes_by_label_Wandell2015
    which were derived from Wandel et al. (2015). These must be provided in a
    'by-label' format (see "by-label data" below).
  * -s<s>|--saturation-constants-by-label=<s>
    Specifies the saturation constants (in the denominator of the divisive
    normalization schema) by visual area. By default, this uses the values in 
    sco.impl.benson17.saturation_constants_by_label_Kay2013, which were derived
    from Kay et al. (2013). These must be provided in a 'by-label' format (see
    "by-label data" below).

Specifying "by-label" Data
================================================================================
Optional arguments that end with "by-label" must be specified in one of three
ways:
  1. A python dictionary whose keys are integers and whose values are floats,
     for example --compressive-constants-by-label='{1:0.9, 2:0.85, 3:0.8}'.
     The keys indicate the visual area id (1/2/3/4 are V1/V2/V3/hV4).
  2. A python list of the values by visual area label in the order of the visual
     area labels. For example, the equivalent to the dict example given in (1)
     is --compressive-constants-by-label='[0.9, 0.85, 0.8]'.
  3. The name of a python variable that contains a valid by-label dictionary,
     for example sco.impl.benson17.compressive_constants_by_label_Kay2013.

Specifying the Gamma Table
================================================================================
The gamma table may be specified with the -g or --gamma options; these may be
used to specify the gamma table in one of two ways:
  1. A "matlab-like" matrix string for an n x 2 matrix. In such a string, the
     rows are separated by semicolons and the columns are separated by commas;
     for example --gamma='[0, 0; 0.25, 0.2; 0.5, 0.38; 0.75, 0.55; 1.0, 1.0]'.
  2. A python list whose values are floats; this is equivalent to the matrix-
     like input in (1); the values in the list are taken to be the second column
     of the matrix and the first column is assumed to be linearly-spaced values
     between 0 and 1; so the above example is equivalent to
      -g'[0, 0.25, 0.38, 0.55, 1]'.
The (given or inferred) matrix specifies a translation of pixel values (first
column) to displayed pixel values (second column); linear interpolation is
performed for pixel values between the provided rows.
'''

def _check_extract(arg, subq=False):
    img_formats = ['.png', '.jpg', '.gif']
    if len(arg) > 7 and arg[-7:] == '.tar.gz' or len(arg) > 4 and arg[-4:] == '.tgz':
        tdir = tempfile.mkdtemp()
        tf = tarfile.TarFile(name=arg, mode='r')
        tf.extractall(path=tdir)
        l = [os.path.join(tdir, f) for f in os.listdir(tdir) if f[0] != '.']
        if subq:
            # Return just the subject path
            if 'surf' in l and 'mri' in l: return tdir
            elif len(l) == 0:
                raise ValueError('Extracted tar-file but found no subject directory!')
            else:
                return os.path.join(tdir, l[0])
        else:
            # Return a list of image paths
            return [fl
                    for fl0 in l
                    for fl in ([fl0] if os.path.isfile(fl0) else
                               [os.path.join(fl0, fff) for fff in os.listdir(fl0)])
                    if len(fl) > 4 and fl[-4:].lower() in img_formats]
    elif subq:
        return arg
    else:
        if os.path.isdir(arg):
            return [os.path.join(arg, ff)
                    for ff in os.listdir(arg)
                    if ff[-4:].lower() in img_formats]
        else:
            return [arg] if len(arg) > 4 and arg[-4:].lower() in img_formats else []

def _parse_label_arg(arg):
    s = ''.join(arg.split()) # eliminate whitespace
    if s[0] == '{' and s.endswith('}'):
        els = s[1:-1].split(',')
        return {int(k):float(v) for el in els for (k,v) in [el.split(':')]}
    elif s[0] == '[' and s.endswith(']'):
        els = s[1:-1].split(',')
        return {(k + 1):float(v) for (k,v) in enumerate(els)}
    else:
        return arg # probably is a variable name...
def _parse_gamma_arg(g):
    if g is None: return None
    s = ''.join(g.split()) # eliminate whitespace
    if s[0] != '[' or s[-1] != ']':
        raise ValueError('gamma argument must be specified as a matrix or vector (in []s)')
    s = s[1:-1]
    rows = s.split(';')
    if len(rows) == 1:
        # we have a vector...
        return np.asarray([float(k) for k in s.split(',')])
    else:
        return np.asarray([[float(c) for c in row.split(',')] for row in rows])
def _parse_measurements_arg(mmfl):
    if ':' in mmfl:
        mmfl = mmfl.split(':')
        if len(mmfl) != 2:
            raise ValueError('surface modality requires 2 :-separated measurement filenames')
        return tuple(mmfl)
    else:
        return mmfl
    
option_parsers = {
    'aperture_edge_width':            float,
    'aperture_radius':                float,
    'background':                     float,
    'gabor_orientations':             int,
    'max_eccentricity':               float,
    'normalized_pixels_per_degree':   float,
    'pRF_n_radii':                    float,
    'gamma':                          _parse_gamma_arg,
    'pRF_sigma_offsets_by_label':     _parse_label_arg,
    'pRF_sigma_slopes_by_label':      _parse_label_arg,
    'divisive_exponents_by_label':    _parse_label_arg,
    'compressive_constants_by_label': _parse_label_arg,
    'contrast_constants_by_label':    _parse_label_arg,
    'saturation_constants_by_label':  _parse_label_arg,
    'gains_by_label':                 _parse_label_arg,
    'measurements_filename':          _parse_measurements_arg}
    
def main(argv):
    (args, opts) = main_parser(argv)
    if opts['help']:
        print sco_help
        return 1
    if len(args) < 3: raise ValueError('Syntax: sco <subject> <pixels-per-degree> <images...>')
    # Arg 0 is a subject; arg 1 is a pixels-per-degree value, rest are images
    sub = _check_extract(args[0], True)
    d2p = float(args[1])
    imfiles = [a for arg in args[2:] for a in _check_extract(arg)]

    # eliminate all the options that weren't specified (allow the sco to determine defaults) and
    # parse the options that require it
    tmp = opts
    opts = {}
    for (k,v) in six.iteritems(tmp):
        if v is Ellipsis: continue
        if k in option_parsers: v = option_parsers[k](v)
        opts[k] = v

    mdl = build_model('benson17')
    r = mdl(opts, subject=sub, stimulus=imfiles, pixels_per_degree=d2p)
    r['exported_files']
    return 0

# Run the main function
sys.exit(main(sys.argv[1:]))
