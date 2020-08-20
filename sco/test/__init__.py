####################################################################################################
# sco/test/__init__.py
# Tests for the standard cortical observer (SCO) library.
# By Noah C. Benson

import unittest, os, sys, six, logging, pimms
import numpy      as np
import scipy      as sp
import pyrsistent as pyr

#from .core import test_image, luminance_grating

class SCOTestArtificial(unittest.TestCase):
    '''
    The SCOTestArtificial class handles tests that generate artificial stimuli, run them through the
    SCO model, and checks that their responses are reasonable.
    '''
    def setUp(self):
        # if the SCO_TEST_REPORT_PATH environment variable is set, we prepare a report of the
        # tests in this directory:
        pp = os.environ.get('SCO_TEST_REPORT_PATH')
        if pp is not None:
            if not os.path.isdir(pp): raise ValueError('SCO_TEST_REPORT_PATH is not a directory')
            self.report_dir = pp
            # if there is a suffix supplied, use it; otherwise use no suffix
            suff = os.environ.get('SCO_TEST_REPORT_SUFFIX')
            if suff is None or suff.lower in ['auto', 'automatic']:
                suff = time.strftime('_%Y-%m-%d-%H:%M:%S', time.localtime())
            # okay, setup the report filename and image directory
            self.report_filename = os.path.join(pp, 'test-report' + suff + '.md')
            self.report_imagedir = os.path.join(pp, 'test-images' + suff)
            if not os.path.isdir(self.report_imagedir): os.makedirs(self.report_imagedir, 0o755)
            # let's give the report file a header
            with open(self.report_filename, 'a') as fl:
                tstr = time.strftime('%Y-%m-%d, %H:%M:%S', time.localtime())
                fl.write('\n# SCO Test Report: ' + tstr + '\n\n')
                fl.write('This report was generated by the sco.test module of the Standard\n')
                fl.write('Cortical Observer Python library.\n')
        else:
            self.report_filename = None
            self.report_imagedir = None
        # okay, next we need to setup the testing sco imap
        from sco.impl.testing import sco_plan
        self.data = sco_plan()
    def report(self, *args):
        '''
        Convert the arguments to a string and append it to the report markdown file.
        '''
        if self.report_filename is None: return None
        with open(self.report_filename, 'a') as fl:
            for arg in args:
                if pimms.is_str(arg):
                    fl.write(arg)
                    fl.write('\n')
                elif pimms.is_vector(arg, str):
                    for s in arg:
                        fl.write(s)
                        fl.write('\n')
                else: raise ValueError('Non-string given to report()')
        return None
    def imsave(self, name, img):
        '''
        Saves the given image with the given name to a file in the report image directory.
        '''
        if self.report_imagedir is None: return None
        from .core import imsave
        if not name.endswith('.png'): name = name + '.png'
        flnm = os.path.join(self.report_imagedir, name)
        imsave(flnm, img)
        return None
    def test_contrast(self):
        pass

if __name__ == '__main__':
    unittest.main()