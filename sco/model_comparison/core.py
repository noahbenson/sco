# core.py
#
# core code for comparing model performances and visualizing them.
#
# By William F Broderick

import argparse
import warnings
import os
import inspect
import re
import pickle
import numpy as np
from scipy import io as sio

# These requirements are optional as far as SCO is concerned, so they are given slightly more
# useful error messages that indicate the library may not be installed.
# --nben 03/06/2018
try:
    import matplotlib.pyplot as plt
except:
    raise RuntimeError('Could not import matplotlib.pyplot; make sure matplotlib is installed')

try:
    import seaborn as sns
except:
    raise RuntimeError('Could not import seaborn; make sure seaborn is installed')

try:
    import pandas as pd
except:
    raise RuntimeError('Could not import pandas; make sure pandas is installed')


# We need to define these before our MODEL_DF_KEYS constant below
def _get_pRF_pixel_centers_row(pRF_views, normalized_stimulus_images, normalized_pixels_per_degree):
    pixel_centers = np.asarray([[list(view._params(im.shape, d2p))[0] for im, d2p in
                                 zip(normalized_stimulus_images, normalized_pixels_per_degree)]
                                for view in pRF_views])
    return pixel_centers[:, :, 0]


def _get_pRF_pixel_centers_col(pRF_views, normalized_stimulus_images, normalized_pixels_per_degree):
    pixel_centers = np.asarray([[list(view._params(im.shape, d2p))[0] for im, d2p in
                                 zip(normalized_stimulus_images, normalized_pixels_per_degree)]
                                for view in pRF_views])
    return pixel_centers[:, :, 1]


def _get_pRF_pixel_size(pRF_views, normalized_stimulus_images, normalized_pixels_per_degree):
    pixel_sizes = np.asarray([[list(view._params(im.shape, d2p))[1] for im, d2p in
                               zip(normalized_stimulus_images, normalized_pixels_per_degree)]
                              for view in pRF_views])
    return pixel_sizes


# Keys from the results dict that correspond to model parameters and so that we want to save in the
# output dataframe
MODEL_DF_KEYS = ['pRF_centers', 'pRF_hemispheres', 'pRF_voxel_indices', 'SOC_responses',
                 'pRF_eccentricity', 'pRF_v123_labels', 'predicted_responses', 'pRF_sizes',
                 'pRF_polar_angle', 'Kay2013_output_nonlinearity', 'Kay2013_pRF_sigma_slope',
                 'Kay2013_SOC_constant', 'Kay2013_normalization_r', 'Kay2013_normalization_s',
                 'Kay2013_response_gain', 'pRF_frequency_preferences', 'voxel_idx',
                 {'pRF_pixel_centers_row': _get_pRF_pixel_centers_row}, 'effective_pRF_sizes',
                 {'pRF_pixel_sizes': _get_pRF_pixel_size},
                 {'pRF_pixel_centers_col': _get_pRF_pixel_centers_col}, 'pRF_blob_stds']
    
# Keys from the results dict that correspond to model setup and so we want
# to save them but not in the dataframe.
MODEL_SETUP_KEYS = ['max_eccentricity', 'normalized_pixels_per_degree', 'stimulus_edge_value',
                    'gabor_orientations', 'stimulus_pixels_per_degree', 'subject',
                    'stimulus_contrast_functions', 'normalized_stimulus_aperture',
                    'pRF_frequency_preference_function', 'stimulus_aperture_edge_width',
                    'normalized_contrast_functions']

# Keys that show the stimulus images in some form.
IMAGES_KEYS = ['stimulus_images', 'normalized_stimulus_images', 'stimulus_image_filenames']

# Keys that show brain data in some form.
BRAIN_DATA_KEYS = ['v123_labels_mgh', 'polar_angle_mgh', 'ribbon_mghs', 'eccentricity_mgh']


def _check_default_keys(results, keys=None):
    if keys is None:
        keys = (MODEL_DF_KEYS + MODEL_SETUP_KEYS + IMAGES_KEYS + BRAIN_DATA_KEYS)
    for k in results:
        if k not in keys:
            warnings.warn("Results key %s is not in any of our default key sets!" % k)
    for k in keys:
        if isinstance(k, basestring):
            if k not in results:
                warnings.warn("Default key %s not in results!" % k)
        # if the key's not a string, we assume it's a dictionary and then we assume its value is a
        # function whose keys are all string that we'll grab from results
        elif isinstance(k, dict):
            assert len(k)==1, "Don't know how to handle more than one entry here!"
            # grab the value corresponding to the first (and only) key
            for arg in inspect.getargspec(k.items()[0][1]).args:
                # if this contains 'idx' then this is an index into the array.
                if 'idx' not in arg:
                    assert arg in results, "Required key %s not in results!" % arg

def create_setup_dict(results, keys=None):
    """Return a dictionary containing just the model setup-relevant keys

    I don't think returning a dataframe makes sense here, because we mainly want these around for
    reference. Keys are determined by the constant MODEL_SETUP_KEYS, which can be overwritten by
    the user using the kwarg keys.
    """
    setup_dict = {}
    if keys:
        setup_keys = keys
    else:
        setup_keys = MODEL_SETUP_KEYS
    for k in setup_keys:
        try:
            setup_dict[k] = results[k]
        except KeyError:
            warnings.warn("Results dict does not contain key %s, skipping" % k)
    return setup_dict

def create_images_dict(results, keys=None):
    """Return a dictionary containing just the keys corresponding to the stimulus images

    I don't think returning a dataframe makes sense here, because we mainly want these around for
    reference. Keys are determined by the constant IMAGES_KEYS, which can be overwritten by the
    user using the kwarg keys.
    """
    images_dict = {}
    if keys:
        images_keys = keys
    else:
        images_keys = IMAGES_KEYS
    for k in images_keys:
        try:
            images_dict[k] = results[k]
        except KeyError:
            warnings.warn("Results dict does not contain key %s, skipping" % k)
    return images_dict

def create_brain_dict(results, keys=None):
    """Return a dictionary containing just the keys corresponding to the input brain images

    I don't think returning a dataframe makes sense here, because we mainly want these around for
    reference. Keys are determined by the constant BRAIN_DATA_KEYS, which can be overwritten by the
    user using the kwarg keys.
    """
    brain_dict = {}
    if keys:
        brain_keys = keys
    else:
        brain_keys = BRAIN_DATA_KEYS
    for k in brain_keys:
        try:
            brain_dict[k] = results[k]
        except KeyError:
            warnings.warn("Results dict does not contain key %s, skipping" % k)
    return brain_dict

def create_model_dataframe(results, image_names, model_df_path="./soc_model_params.csv", keys=None,
                           image_num=None, voxel_num=None):
    """creates the model dataframe from the results dict

    This function takes in a results dict create by one sco_chain run and turns it into a pandas
    dataframe for further analysis or export. The dataframe *does not* contain all of the items in
    results. It only contains those items that are necessary for calculating the output of the
    model. Those keys are given by the constant MODEL_DF_KEYS and can be overwritten by the user
    using the kwarg keys.

    We need a list / array image_names, which gives the names of each of the images, in order. This
    way we can give them more informative labels (even if it's just the indexes in a matlab array,
    as when using Kay2013's stimuli.mat) for direct comparison. the values of image_names can
    either be all integers or all strings, the two cannot mix.

    We also assume that this results dictionary will contain the key 'predicted_responses', which
    is the final output of the model. We assume that this is an array of voxels x images, and so
    infer the number of voxels and images that this model was fit on from this array. If for some
    reason this is not true (which it really should be -- you will basically always have
    predicted_responses), then set image_num and voxel_num to the number of images and voxels,
    respectively.

    Arguments
    =============================

    model_df_path: string. Absolute path to save the model dataframe at. By default, will be saved
    as soc_model_params.csv in the current directory.
    """
    model_df_dict = {}
    if voxel_num and image_num:
        warnings.warn("Using user-set voxel_num and image_num instead of inferring from data")
    elif (voxel_num and not image_num) or (image_num and not voxel_num):
        raise Exception("image_num and voxel_num must be both set or both None, you cannot set"
                        " only one!")
    else:
        image_num, voxel_num = results['predicted_responses'].shape
    if keys:
        model_keys = keys
    else:
        model_keys = MODEL_DF_KEYS
    _check_default_keys(results, model_keys+MODEL_SETUP_KEYS+IMAGES_KEYS+BRAIN_DATA_KEYS)
    for k in model_keys:
        if isinstance(k, dict):
            # we've already asserted that there's only one entry in k.
            k, tmp_func = k.items()[0]
            tmp_args = inspect.getargspec(tmp_func).args
            value = tmp_func(*[results.get(i) for i in tmp_args])
        else:
            value = results[k]
        try:
            # they should all be arrays, but just in case.
            model_df_dict[k] = np.asarray(value)
        except KeyError:
            warnings.warn("Results dict does not contain key %s, skipping" % k)
    # We assume that we have four types of variables here, as defined by their shape:
    #
    # 1. One-dimensional. For these, we simply want their values to be one of the columns of our
    # dataframe. Example: pRF_v123_labels, where each value says whether the corresponding voxel is
    # in V1, V2 or V3.
    #
    # 2. Two-dimensional, voxels on the first dimension, second dimension doesn't correspond to
    # images. For these, we just assume that each column of the array should be a separate column
    # in our dataframe (with its name taken from the name of the variable with a suffix of '_0',
    # '_1' etc ). Example: pRF_centers, where each value gives the x and y coordinates of the
    # center of the voxel's pRF.
    #
    # 3. Two-dimensional, voxels on the first dimension, images on the second or similarly, images
    # on the first and voxels on the second dimensions. For these, we assume these variables give
    # the response (of some kind) to each image of each voxel. Thus each row should be a separate
    # column in our dataframe, suffixed with '_image_1', '_image_2', etc. These will correspond to
    # the names in the image dataframe. Example: predicted response, where each value gives the
    # model's predicted response of that voxel to the given image.
    # 
    # 4. Three-dimensional, with voxels on the first dimension, images on the second, and something
    # else on the third; we give each of those third dimensions a separate column similar to case 2
    # above and each of the second dimensions a separate column similar to case 3. So if the array
    # is voxels x 2 x 2, each voxel will have four columns: _image_0_0, image_0_1, image_1_0, and
    # _image_1_1. Example: pRF_pixel_centers, where each value gives the x and y coordinates of the
    # center of each voxel's pRF in pixels for a given image (since images can have different
    # pixels per degrees, this isn't the same for every image).
    #
    # As we loop through these variables, we split up any two dimensional arrays so that each value
    # in the dictionary is a one-dimensional array, all with length equal to the number of
    # voxels. This will make the creation of the dataframe very easy.
    # 
    # Entries in MODEL_DF_KEYS can also be dictionaries, in which case they must have one key,
    # value pair and the key is the column in the dataframe we'll place it in and the value is a
    # function that takes keys from results as its args and returns an array that fits in one of
    # the categories above.

    # We use this tmp_dict to construct the model_df
    tmp_dict = {}
    if isinstance(image_names[0], int):
        # if image_names contains integers, then we use this %04d string formatting
        img_format_string = "%s_image_%04d"
    else:
        # else we just use the values as is.
        img_format_string = "%s_image_%s"
    for k, v in model_df_dict.iteritems():
        if len(v.shape) == 1 or len(v.shape)==0:
            # This is case 1, and we grab the value as is (or there's only one value, so it's the
            # same for all)
            tmp_dict[k] = v
        elif len(v.shape) == 2:
            if v.shape[0] == voxel_num:
                if v.shape[1] != image_num:
                    # then this is case two above.
                    for i in range(v.shape[1]):
                        tmp_dict["%s_dim%s" % (k, i)] = v[:, i]
                else:
                    # then this is case three above
                    for i in range(v.shape[1]):
                        tmp_dict[img_format_string % (k, image_names[i])] = v[:, i]
            elif v.shape[0] == image_num:
                if v.shape[1] != voxel_num:
                    raise Exception("For variable %s, images are on the first dimension but voxels"
                                    " not on the second! (dimensions: %s)" % (k, v.shape))
                # then this is case three above
                for i in range(v.shape[0]):
                    tmp_dict[img_format_string % (k, image_names[i])] = v[i, :]
            else:
                raise Exception("Result variable %s is two dimensional but the first dimension "
                                "doesn't correspond to the number of voxels or images (dimensions:"
                                " %s)!" % (k, v.shape))
        elif len(v.shape) == 3:
            # Here, we assume it's voxels x images x something, like case four above.
            if v.shape[0] == voxel_num:
                for i in range(v.shape[1]):
                    for j in range(v.shape[2]):
                        tmp_dict[(img_format_string + "_dim%s") % (k, image_names[i], j)] = v[:, i, j]
            else:
                raise Exception("Result variable %s is three dimensional but the first dimension "
                                "doesn't correspond to the number of voxels (dimensions: %s)!" %
                                (k, v.shape))
        else:
            raise Exception('Result variable %s is not a one, two, or three dimensional array'
                            ' (shape %s)!' % (k, v.shape))
    model_df = pd.DataFrame(tmp_dict)
    # This isn't necessary, but I find it easier to examine if the columns are sorted
    # alphabetically
    model_df = model_df.reindex_axis(sorted(model_df.columns, key=lambda x: x.lower()), axis=1)
    # I prefer the letters, I think that makes it clearer.
    if 'pRF_hemispheres' in model_df.columns:
        model_df.pRF_hemispheres = model_df.pRF_hemispheres.map({1: 'L', -1: 'R'})
    # if we've included voxel_idx, make that (with the name of voxel) the index. Otherwise, we
    # infer voxel label from the indices directly.
    if 'voxel_idx' in model_df.columns:
        model_df = model_df.rename(columns={'voxel_idx': 'voxel'})
        model_df = model_df.set_index('voxel')
    # Finally, we save model_df as a csv for easy importing / exporting
    model_df.to_csv(model_df_path, index_label='voxel')
    sio.savemat(os.path.splitext(model_df_path)[0] + "_image_names.mat",
                {'image_names': image_names})
    return model_df


def _load_pkl_or_mat(path, mat_field):
    if os.path.splitext(path)[1] == ".mat":
        path = sio.loadmat(path)[mat_field]
    elif os.path.splitext(path)[1] == ".pkl":
        with open(path) as f:
            path = pickle.load(f)
    else:
        raise Exception("Don't know how to handle extensions %s" %
                        os.path.splitext(path)[1])
    return path


def _create_plot_df(model_df, stimuli_idx=None, stimuli_descriptions=None):
    """create dataframe stimuli associated with condition to plot from

    condition can be either a boolean array, in which case we grab the values from stimuli_idx
    corresponding to the Trues, or an array of integers, in which case we assume it's just the
    index numbers themselves.
    """
    # this grabs the columns that contain the predicted responses from both the matlab and the
    # python code.
    plot_df = model_df.filter(like="predicted_responses_image")

    plot_df = plot_df.reset_index().rename(columns={'index':'voxel'})
    plot_df = pd.wide_to_long(plot_df, ["predicted_responses", "MATLAB_predicted_responses"],
                              i='voxel', j='image')
    plot_df = plot_df.rename(columns={'predicted_responses':'predicted_responses_languagepython',
                                      'MATLAB_predicted_responses': 'predicted_responses_languageMATLAB'})
    plot_df = plot_df.reset_index()
    plot_df = pd.wide_to_long(plot_df, ['predicted_responses'], i='voxel', j='language')
    plot_df = plot_df.reset_index()

    plot_df['language'] = plot_df['language'].apply(lambda x: x.replace('_language', ''))
    plot_df['image'] = plot_df['image'].apply(lambda x: x.replace('_image_', ''))

    # right now, this only works when they all have subimages or when none of them do. Should
    # probably fix this eventually.
    if 'sub' in plot_df['image'].iloc[0]:
        plot_df['subimage'] = plot_df['image'].apply(lambda x: re.search(r'[0-9]*_sub([0-9]*)', x).groups()[0])
        plot_df['image'] = plot_df['image'].apply(lambda x: re.search(r'([0-9]*)_sub[0-9]*', x).groups()[0])

    if stimuli_descriptions is not None:
        mapping = dict(('%04d' % k, v) for k, v in zip(stimuli_idx, stimuli_descriptions))
        plot_df['image_name'] = plot_df.image.map(mapping)

    plot_df = plot_df.set_index('voxel')
    plot_df['v123_label'] = model_df['pRF_v123_labels']
    plot_df = plot_df.reset_index()

    return plot_df


def _plot_stimuli(condition, stimuli_idx, stimuli, stimuli_descriptions, results=None,
                  model_df=None, stimulus_model_names=None, subflag=False):
    """plot stimuli associated with condition

    condition can be either a boolean array, in which case we grab the values from stimuli_idx
    corresponding to the Trues, or an array of integers, in which case we assume it's just the
    index numbers themselves.

    subflag, optional, can be False or an integer. It's used to examine the 'subimages' of our
    stimuli: several stimuli have multiple frames in them which were presented together; we refer
    to these frames as the 'subimages'. If subflag is false, then we show the first subimage of
    each of the indices. Else, subimage should be an index into the images we're showing and we'll
    then show all the subimages for that stimulus. (e.g., condition = [139, 140, 141] and subflag =
    0; then we'll show all the subimages of stimulus 139).
    """
    if isinstance(condition, basestring):
        condition = [condition == description for description in stimuli_descriptions]
        tmp_idx = stimuli_idx[np.where(condition)]
    else:
        # assume it's just the index numbers directly
        tmp_idx = np.asarray(condition)
    fig = plt.figure(figsize=[10, 10])
    if results is None:
        if not subflag:
            # we want to check whether vmax should be 1 or 255. to do that, we just check if any
            # values in the first image we're using are greater than 1 (if vmax is supposed to be
            # 255, each image should have values greater than 1)
            if np.any(stimuli[tmp_idx[0]]>1):
                vmax = 255
            else:
                vmax = 1
            for i, idx in enumerate(tmp_idx):
                ax = fig.add_subplot(np.ceil(np.sqrt(len(tmp_idx))), np.ceil(np.sqrt(len(tmp_idx))),
                                     i+1)
                plt.imshow(stimuli[idx][:, :, 0], cmap='gray', vmin=0, vmax=vmax)
                plt.title((idx, 0))
                ax.xaxis.set_visible(False)
                ax.yaxis.set_visible(False)
        else:
            if np.any(stimuli[tmp_idx[subflag[0]]]>1):
                vmax = 255
            else:
                vmax = 1
            for idx in range(stimuli[tmp_idx[subflag]].shape[2]):
                ax = fig.add_subplot(np.ceil(np.sqrt(stimuli[tmp_idx[subflag]].shape[2])),
                                     np.ceil(np.sqrt(stimuli[tmp_idx[subflag]].shape[2])), idx+1)
                plt.imshow(stimuli[tmp_idx[subflag]][:, :, idx], cmap='gray', vmin=0, vmax=vmax)
                plt.title((tmp_idx[subflag], idx))
                ax.xaxis.set_visible(False)
                ax.yaxis.set_visible(False)
    else:
        norm_stim = results['normalized_stimulus_images']
        vox_num = model_df.shape[0]
        vox_colors = sns.palettes.color_palette('Set1', vox_num)
        if np.any(norm_stim>1):
            vmax = 255
        else:
            vmax = 1
        for i, idx in enumerate(tmp_idx):
            ax = fig.add_subplot(np.ceil(np.sqrt(len(tmp_idx))), np.ceil(np.sqrt(len(tmp_idx))),
                                 i+1)
            # because of how np.where works, this will return a tuple with one entry: an array of
            # length 1. For this to work, we need the index to be an integer, so we just grab the
            # value. np.squeeze will do that as best it can. If there's more than one value here
            # somehow, an error will be thrown when we try to show the image.
            stim_idx = np.squeeze(np.where(stimulus_model_names=='%04d_sub00' % idx))
            plt.imshow(norm_stim[stim_idx, :, :], cmap='gray', vmin=0, vmax=vmax)
            plt.title(stimulus_model_names[stim_idx])
            ax.xaxis.set_visible(False)
            ax.yaxis.set_visible(False)
            circles = []
            for vox_idx in range(vox_num):
                circles.append(plt.Circle(
                    (int(model_df['pRF_pixel_centers_col_image_%s'%stimulus_model_names[stim_idx]][vox_idx]),
                     int(model_df['pRF_pixel_centers_row_image_%s'%stimulus_model_names[stim_idx]][vox_idx])),
                    int(results['effective_pRF_sizes'][vox_idx]*results['normalized_pixels_per_degree'][stim_idx]),
                    color=vox_colors[vox_idx], fill=False))
                ax.add_artist(circles[-1])
        fig.legend(circles, ['vox %d' % i for i in range(vox_num)])
    return fig


def visualize_model_comparison(conditions, condition_titles, model_df, stimulus_model_names,
                               stimuli_descriptions, stimuli, stimuli_idx, plot_kwargs=None,
                               draw_pRF_flag = False,
                               image_template_path="./SCO_model_comparison_{condition}_{plot_type}.svg"):
    """visualize the model comparisons

    Arguments
    ==================

    conditions: list of conditions to plot. Each should be either a string (in which case we plot
    the predictions for those stimuli whose description contain that string) or a list of integers
    (in which case we plot the predictions for the stimuli with those indices).

    condition_titles: list of strings. Should be the same length as conditions and be the title we
    want to associate with those plots. Will title the corresponding graphs and go into their
    filename.

    model_df: pandas dataframe or a string with the path to the csv of that dataframe. This should
    contain the results of running the model using both the python and accompanying matlab code
    (and so should have both columns for both predicted_responses and MATLAB_predicted_responses)

    stimulus_model_names: 1d numpy array or string with path to that array (can be .mat or .pkl; ;
    if it's a .mat then we assume the structure that contains this is called "image_names"). This
    should contain the names of the images the model generated predictions for. This array is
    created by and saved by compare_with_Kay2013.main and these names show up in the columns of the
    model_df.

    stimuli_descriptions: 1d numpy array or string with path to that array (can be .mat or .pkl; if
    it's a .mat then we assume the structure that contains this is called "stimuliNames"). This
    should contain the descriptions of the stimuli used by the model (Catherine Olsson created this
    and it should be found in this repository), and is used to select which images we want to plot
    the comparisons for.

    stimuli: 1d numpy array of 2d and 3d numpy arrays or string with path to that array (can be
    .mat or .pkl). This is the superset of images that the model was trained on (ie, can include
    more than the images used). Should have the same length as stimuli_descriptions.

    stimuli_idx: 1d numpy array or None. Specifies which of the stimuli (in the stimuli array) the
    model were trained on. If all were used, this should be None.

    plot_kwargs: dictionary or list of dictionaries (same length as conditions), optional. Plot
    arguments (passed to sns.factorplot) to use when plotting the data corresponding to each
    condition. If only one dictionary, will use the same for all. Otherwise, condition[i] will be
    plotted with plot_kwargs[i]

    image_template_path: string. String to save the resulting plots at. Should contain {condition}
    (which will be filled with the corresponding condition title) and {plot_type} (which will be
    filled with "stimuli", showing the corresponding stimuli, and "predictions", showing the
    corresponding predictions).
    """
    if isinstance(model_df, basestring):
        # in this case, we also need to grab the results dict
        if draw_pRF_flag:
            results_path = os.path.splitext(model_df)[0] + "_results_dict.pkl"
            # the model path may contain "MATLAB_", because it was run through matlab. But the
            # results dictionary path never will, because it's saved after the python run.
            results_path = results_path.replace("MATLAB_", "")
            with open(results_path) as f:
                results = pickle.load(f)
        model_df = pd.read_csv(model_df)

    if isinstance(stimulus_model_names, basestring):
        stimulus_model_names = _load_pkl_or_mat(stimulus_model_names, 'image_names')

    if stimuli_idx is not None:
        stimuli_idx = np.asarray(stimuli_idx)

    if isinstance(stimuli_descriptions, basestring):
        stimuli_descriptions = _load_pkl_or_mat(stimuli_descriptions, 'stimuliNames')
        if stimuli_idx is not None:
            stimuli_descriptions = stimuli_descriptions[0, stimuli_idx]
            stimuli_descriptions = np.asarray([i[0] for i in stimuli_descriptions])

    if isinstance(stimuli, basestring):
        stimuli = _load_pkl_or_mat(stimuli, 'images')
        stimuli = stimuli[0, :]

    if plot_kwargs is None:
        plot_kwargs = [{} for i in conditions]
    elif isinstance(plot_kwargs, dict):
        plot_kwargs = [plot_kwargs for i in conditions]

    for cond, title, kw in zip(conditions, condition_titles, plot_kwargs):
        plot_df = _create_plot_df(model_df, stimuli_idx, stimuli_descriptions)
        
        if isinstance(cond, basestring):
            plot_df = plot_df[plot_df.image_name==cond]
            order = None
        else:
            plot_df = plot_df[[img in ['%04d'%i for i in cond] for img in plot_df.image]]
            order = ['%04d' % i for i in cond]

        # defualt plotting keywords
        if 'hue' not in kw:
            kw['hue'] = 'language'
        if 'col' not in kw:
            kw['col'] = 'voxel'
        if 'col_wrap' not in kw:
            kw['col_wrap'] = 3
        if 'size' not in kw:
            kw['size'] = 8

        g = sns.factorplot(data=plot_df, y='predicted_responses', x='image',
                           legend_out=True, order=order, **kw)
        g.fig.suptitle(title)
        g.fig.subplots_adjust(top=.9, right=.9)
        g.set_xticklabels(rotation=45)
        g.savefig(image_template_path.format(plot_type="predictions",
                                             condition=title.replace(' ', '_')))

        if draw_pRF_flag:
            fig = _plot_stimuli(cond, stimuli_idx, stimuli, stimuli_descriptions, results, model_df,
                                stimulus_model_names)
        else:
            fig = _plot_stimuli(cond, stimuli_idx, stimuli, stimuli_descriptions)
        fig.suptitle(title)
        fig.savefig(image_template_path.format(plot_type="stimuli",
                                               condition=title.replace(' ', '_')))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=("Visualize model predictions stored in model_df, comparing matlab and python "
                     "versions of the model. If called from command line, plots default selection "
                     "of images; for more control, see Compare_with_Kay2013.ipynb"))
    parser.add_argument("mode", help="full or sweep. This tells this block how to plot the results.")
    parser.add_argument("model_df_path", help="string. Path where model dataframe is saved.")
    parser.add_argument("stimulus_model_names",
                        help=("string. Path where names of stimuli model made predictions for are"
                              " saved. Will correspond to the columns of the model_df"))
    parser.add_argument("stimuli_descriptions",
                        help="string. Path where stimuli descriptions are saved.")
    parser.add_argument("stimuli", help="string. Path to stimuli.mat")
    parser.add_argument("stimuli_idx", nargs='+', type=int,
                        help="list of ints. Which indices in the stimuli to run.")
    args = parser.parse_args()
    if args.mode == 'full':
        conditions = ['grating_ori', 'grating_contrast', 'plaid_contrast', 'circular_contrast',
                      [180, 181, 182, 84, 183], range(131, 138), range(69, 100), range(100, 131),
                      range(131, 158)+[180, 181, 182, 84, 183]]
        condition_titles = ['orientations', 'gratings', 'plaid', 'circular', 'sparse', 'size',
                            'horizontal sweep', 'vertical sweep','full']
        plot_kwargs = [{}, {}, {}, {}, {}, {}, {}, {},
                       {'hue': 'image_name', 'col': 'language', 'col_wrap': None}]
        draw_pRF_flag = False
    elif args.mode == 'sweep':
        conditions = [args.stimuli_idx]
        condition_titles = ['sweep']
        plot_kwargs = [{}]
        draw_pRF_flag = True
    visualize_model_comparison(conditions, condition_titles, args.model_df_path,
                               args.stimulus_model_names, args.stimuli_descriptions, args.stimuli,
                               args.stimuli_idx, plot_kwargs, draw_pRF_flag)
