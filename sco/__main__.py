####################################################################################################
# __main__.py
# The main function, if sco is invoked directly as command.
# By Noah C. Benson

import os, sys, math, tempfile, tarfile
import pysistence

from neuropythy.util import CommandLineParser
from sco import (build_model)

main_parser = CommandLineParser(
    [('h', 'help',      'help',              False),
     ('p', 'deg2px',    'pixels_per_degree', '24'),
     ('e', 'max-eccen', 'max_eccentricity',  '20'),
     ('o', 'output',    'output_dir',        '.')])

sco_help = \
'''
Usage: sco <subject> <image0000> <image0001>...
Runs the Standard Cortical Observer prediction routine and exports a series of
MGZ files: prediction_0000.mgz, prediction_0001.mgz, etc.
The following options may be given:
  * -h|--help prints this message.
  * -p|--deg2px=<value> sets the pixels per degree in the input images
    (default: 24).
  * -e|--max-eccen=<value> sets the maximum eccentricity modeled in the output
    (default: 20).
  * -o|--output=<directory> sets the output directory (default: .).
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

def main(argv):
    (args, opts) = main_parser(argv)
    if opts['help']:
        print sco_help
        return 1
    if len(args) < 2: raise ValueError('Syntax: sco <subject> <images...>')
    opts['pixels_per_degree'] = float(opts['pixels_per_degree'])
    opts['max_eccentricity']  = float(opts['max_eccentricity'])
    # Arg 0 is a subject; arg 1 is a directory of images
    sub = _check_extract(args[0], True)
    imfiles = [a for arg in args[1:] for a in _check_extract(arg)]
    mdl = build_model('benson17')
    r = mdl(opts, subject=sub, stimulus_image_filenames=imfiles)
    r['exported_files']
    return 0

# Run the main function
sys.exit(main(sys.argv[1:]))
