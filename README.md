# segpc

INSTRUCTIONS TO REPRODUCE XLAB INSIGHTS' RESULTS IN SEGPC-2021 COMPETITION

--------------------------------------------------------------------------------

WE PROVIDE:

- A file 'environment.yml' ready to create a conda environment to be able to run the provided code. To do so run: conda env create -f environment.yml

- A directory 'final_ensemble_config_files' with seven MMDETECTION (https://github.com/open-mmlab/mmdetection) configuration files used to train the individual models that were included in the winning final ensemble. In order to retrain the models, MMDETECTION (https://github.com/open-mmlab/mmdetection) and MMCV (https://github.com/open-mmlab/mmcv) are needed. Please refer to https://github.com/open-mmlab/mmdetection/blob/master/docs/get_started.md for installation instructions.

- A directory 'code' with three files:
  - An iPython notebook 'single_model.ipynb' used to generate prediction images from raw predictions generated in JSON format by MMDETECTION. This notebook can be used as well to generate pandas DataFrames that are later used in the ensemble notebook ('ensemble.ipynb'). Both raw predictions and DataFrames are provided directly by us too, so there's no need to re-run this notebook.
  - An iPython notebook 'ensemble.ipynb' that can be used to generate the final images submitted to the competition. 
  - A file 'utils.py' with some functions needed by the notebooks. 

- This README.md file.

--------------------------------------------------------------------------------

IN ORDER TO GENERATE ALL THE IMAGES THAT COMPOSED THE FINAL WINNING SUBMISSION OF SEGPC-2021, PLEASE CREATE A CONDA ENVIRONMENT USING THE PROVIDED 'environment.yml' FILE AND EXECUTE (WITHIN THE CREATED ENVIRONMENT) THE PYTHON NOTEBOOK 'ensemble.ipynb'. PLEASE, SPECIFY THE REQUIRED VARIABLES IN THE 'Execution parameters' CELL.

--------------------------------------------------------------------------------

Please contact alvaro.garcia.faura@xlab.si in case any clarification is needed.

XLAB Insights.