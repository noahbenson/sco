SHELL := /bin/bash

# for all full stimuli, use the following:
STIMULI_IDX_full = $(shell seq 69 224)

# for all sweep stimuli, use the following (and honestly, this is
# small enough that you don't need to use something else for testing)
STIMULI_IDX_sweep = $(shell seq 0 33)

# for testing:
# STIMULI_IDX_full := $(shell seq 69 73)

VOXEL_IDX = $(shell seq 0 9)

# KNK_PATH=/home/billbrod/Documents/Kendrick-socmodel/code/
KNK_PATH=/Users/winawerlab/matlab/git/knkutils/
SUBJ=test-sub
SUBJ_DIR=/Volumes/server/Freesurfer_subjects
# SUBJ_DIR=/home/billbrod/Documents/SCO-test-data/Freesurfer_subjects

# make sure matlab is in your path, which it may not be by default if you're on Mac.

# for our stimuli, we use the pictures from Kay2013, which Kendrick
# provides on his website.
full_stimuli.mat : 
	wget -q http://kendrickkay.net/socmodel/stimuli.mat -O ./full_stimuli.mat
        # we need to do this to get the stimuli.mat into the format we want
	matlab -nodesktop -nodisplay -r "load('$@','images'); save('$@','images'); quit"

sweep_stimuli.mat : full_stimuli.mat
	python2.7 pRF_check.py $< {}_stimuli.mat

# this will also create soc_model_params_%_image_names.mat in the same call
soc_model_params_%.csv : %_stimuli.mat
	python2.7 model_comparison_script.py $< $(SUBJ) $@ $(STIMULI_IDX_$*) -v $(VOXEL_IDX) -s $(SUBJ_DIR)



Kay2013_comparison/MATLAB_soc_model_params_%.csv : Kay2013_comparison/soc_model_params_%.csv
        # we increment the stimuli index and not the voxel index,
        # because the voxel indices refer to a column in the
        # dataframe/table, while the stimuli indices will actually be
        # used to grab something from an array in matlab
	matlab -nodesktop -nodisplay -r "cd $(shell pwd)/sco/model_comparison; compareWithKay2013('$(KNK_PATH)', '$(shell pwd)/Kay2013_comparison/$*_stimuli.mat', [$(STIMULI_IDX_$*)]+1, [$(VOXEL_IDX)], '$(shell pwd)/$<', '$(shell pwd)/Kay2013_comparison/soc_model_params_$*_image_names.mat', '$(shell pwd)/$@'); quit;"

.PHONY : %_images
# this will create several images, with names based on the default options in sco/model_comparison/core.py
%_images : MATLAB_soc_model_params_%.csv %_stimuli.mat soc_model_params_%.csv
	python2.7 sco/model_comparison/core.py $* $< soc_model_params_$*_image_names.mat sco/model_comparison/stimuliNames.mat $*_stimuli.mat $(STIMULI_IDX_$*)


.PHONY : %clean
%clean : 

# this way these won't be deleted as unnecessary intermediates. These
# take a while to make, so we don't want to do that.
.PRECIOUS : soc_model_params_%.csv MATLAB_soc_model_params_%.csv
