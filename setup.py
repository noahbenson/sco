#! /usr/bin/env python

from setuptools import setup

setup(
    name='sco',
    version='0.3.7',
    description='Library for the prediction of cortical responses to visual stimuli: the Standard Cortical Observer',
    keywords='neuroscience vision cortex ',
    author='Noah C. Benson',
    author_email='nben@nyu.edu',
    url='https://github.com/winawerlab/sco/',
    license='GPLv3',
    packages=['sco',
              'sco.util',
              'sco.anatomy',
              'sco.stimulus',
              'sco.pRF',
              'sco.contrast',
              'sco.analysis',
              'sco.impl',
              'sco.impl.benson17',
              'sco.impl.kay13',
              'sco.model_comparison'],
    package_data={'': ['LICENSE.txt']},
    install_requires=['numpy        >= 1.2.0',
                      'scipy        >= 0.7.0',
                      'nibabel      >= 2.0.0',
                      'pyrsistent   >= 0.12.0',
                      'neuropythy   >= 0.4.11',
                      'pimms        >= 0.2.5',
                      'scikit-image >= 0.13.1'])
                      #'matplotlib  >= 1.5.0',
                      #'pandas      >= 0.18.1',
                      #'seaborn     >= 0.7.1'])
