# sco ##############################################################################################
The Standard Cortical Observer (SCO) Python library.

## Authors #########################################################################################

 * [Noah C. Benson](https://github.com/noahbenson) (*corresponding author*) &lt;<nben@nyu.edu>&gt;
 * [Bill Broderick](https://github.com/billbrod)
 * [Catherine Olson](https://github.com/catherio)
 * [Heiko Mueller](https://github.com/heikomuller)
 * [Jonathan Winawer](https://github.com/jwinawer)


## Introduction ####################################################################################

The Standard Cortical Observer (SCO) library is intended as a both a model and modeling framework
for predicting cortical responses to visual stimuli. The library can be run directly or called from
Python code, and it is designed such that model comparison and modification are easy.

### Dependencies ###################################################################################
The Standard Cortical Observer library depends on several libraries, all freely available and listed
in the setup.py and requirements.txt files. In order to install them (assuming they weren't
installed by pip when you install the SCO itself), you can type `pip install -r requirements.txt`.

### Usage ##########################################################################################
The SCO library can be used primarily in two different ways: from the command line or as a library
from Python code. Both of these sections assume that you have installed the SCO library in a
location that is accessibly to Python (e.g., you installed it via pip or you put it on your
PYTHONPATH environment variable).

#### Command Line Usage ############################################################################
To call the SCO from the command line, you can invoke it `__main__` mathod via Python:

```bash
> python -m sco.__main__ --help
```

Currently, it is only possible to change a few options to the model via the command-line; these are
generally the options that go hand-in-hand with the input or outputs.

#### Calling the Library Directly ##################################################################
A better way to call the SCO library is to invoke it via Python. To do this, you can import the sco
package then build a model and query it:
```python
import sco

# Build the Benson17 version of the model:
mdl = sco.build_model('benson17')

# Ask it what its afferent parameters are:
mdl.afferents
# => ('contrast_constants_by_label', 'aperture_radius', 'create_directories',
# =>  'max_eccentricity', 'measurements_filename', 'modality', 'subject',
# =>  'divisive_exponents_by_label', 'pixels_per_degree',
# =>  'pRF_sigma_slopes_by_label', 'output_suffix',
# =>  'divisive_normalization_schema', 'pRF_n_radii',
# =>  'compressive_constants_by_label', 'output_directory', 'background',
# =>  'normalized_pixels_per_degree', 'gabor_orientations',
# =>  'aperture_edge_width', 'stimulus', 'output_prefix', 'import_filter',
# =>  'cpd_sensitivity_function', 'saturation_constants_by_label', 'gamma')

# Ask it which parameters have default values (and what they are):
# (note that a pmap is a kind of persistent dict object)
mdl.defaults
# => pmap({'contrast_constants_by_label': 'sco.impl.benson17.contrast_constants_by_label_Kay2013',
# =>       'normalized_pixels_per_degree': 12, 'gabor_orientations': 8,
# =>       'pRF_n_radii': 3.0, 'aperture_radius': None, 'output_prefix': '',
# =>       'gamma_correction_function': None, 'aperture_edge_width': None,
# =>       'background': 0.5, 'modality': 'volume', 'gamma': None
# =>       'compressive_constants_by_label': 'sco.impl.benson17.compressive_constants_by_label_Kay2013',
# =>       'divisive_exponents_by_label': 'sco.impl.benson17.divisive_exponents_by_label_Kay2013',
# =>       'pRF_sigma_slopes_by_label': 'sco.impl.benson17.pRF_sigma_slopes_by_label_Kay2013',
# =>       'divisive_normalization_schema': 'Heeger1992',
# =>       'cpd_sensitivity_function': 'sco.impl.benson17.cpd_sensitivity',
# =>       'saturation_constants_by_label': 'sco.impl.benson17.saturation_constants_by_label_Kay2013',
# =>       'max_eccentricity': 12, 'output_suffix': '', 'import_filter': None,
# =>       'create_directories': False})

# Ask it about one of the parameters
print mdl.afferent_docs['subject']
# => (import_subject) subject: Must be one of (a) the name of a FreeSurfer subject found on the subject path,
# =>         (b) a path to a FreeSurfer subject directory, or (c) a neuropythy FreeSurfer subject
# =>         object.

# Ask it what outputs it produces:
mdl.efferents.keys()
# => pvector(['prediction_analysis', 'pRF_SOC', 'freesurfer_subject',
# =>          'stimulus_ordering', 'contrast_constants', 'contrast_energies',
# =>          'exported_report_filenames', 'coordinates', 'measurements',
# =>          'pRFs', 'divisive_normalization_parameters', 'hemispheres',
# =>          'pRF_sigma_slopes', 'compressive_constants',
# =>          'measurement_hemispheres', 'pRF_radii',
# =>          'exported_analysis_filenames', 'measurement_per_prediction',
# =>          'pRF_sigmas', 'cortex_coordinates', 'cortex_indices',
# =>          'polar_angles', 'prediction', 'prediction_per_measurement',
# =>          'cpd_sensitivities', 'pixel_centers', 'image_names',
# =>          'divisive_normalization_function', 'contrast_filter',
# =>          'stimulus_map', 'measurement_indices', 'image_array',
# =>          'exported_files', 'exported_predictions_filenames',
# =>          'benson17_default_options_used', 'gamma_correction_function',
# =>          'eccentricities', 'labels', 'corresponding_indices',
# =>          'pRF_centers', 'cortex_affine', 'prediction_analysis_labels',
# =>          'measurement_coordinates'])

# Ask for help about of of the ouptuts:
print mdl.efferent_docs['prediction']
# => (compressive_nonlinearity) prediction: Will be the final predictions of %BOLD-change for each pRF examined, up to gain.
# =>         The data will be stored in an (n x m) matrix where n is the number of pRFs (see labels,
# =>         hemispheres, cortex_indices) and m is the number of images.

# Results may be generated by providing the model with parameters; parameters
# may be provided as a dict of key-value pairs (optional) followed by any number
# of keyword arguments:
results = mdl({'subject': 'test-sub-01', 'max_eccentricity': 10},
              pixels_per_degree=6, stimulus=stimulus_file_list))

# results is now a dict-like object; the keys are the efferent (output) data listed above;
# the values do not get calculated until you request them, so while the above line that
# begins with "results = mdl(..." return immediately, the following line will take some
# time to run:
results['prediction'].shape #should be vox-by-imgs
# => (17551, 120)

# The following like will force all exported files in the standard 'benson17' model to be
# exported:
results['exported_files']
# => <list of filenames for exported prediction, images, reports, etc.>
```

## License #########################################################################################

This README file is part of the SCO Python library.

The SCO Python library is free software: you can redistribute it
and/or modify it under the terms of the GNU General Public License as
published by the Free Software Foundation, either version 3 of the
License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.


## References

* Kay KN, Winawer J, Rokem A, Mezer A, Wandell BA (**2013**) A two-stage
  cascade model of BOLD responses in human visual cortex. *PLOS Comput. Biol.*
  **9**(5):1003079. doi:10.1371/journal.pcbi.1003079
* Kay KN, Winawer J, Mezer A, Wandell BA (**2013**). Compressive spatial
  summation in human visual ortex. *J. Neurophysiol.* **110**(2):481–494.
  doi:10.1152/jn.00105.2013
* Benson NC, Butt OH, Brainard DH, Aguirre GK (**2014**) Correction of
  distortion in flattened representations of the cortical surface allows
  prediction of V1-V3 functional organization from anatomy. *PLOS Comput.
  Biol* **10**(3):e1003538. doi:10.1371/journal.pcbi.1003538
* Benson NC, Butt OH, Datta R, Radoeva PD, Brainard DH, Aguirre GK (**2012**)
  The retinotopic organization of striate cortex is well predicted by surface
  topology. *Curr. Biol.* **22**(21):2081-5. doi:10.1016/j.cub.2012.09.014.