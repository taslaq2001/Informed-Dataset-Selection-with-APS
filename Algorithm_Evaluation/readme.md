### APS Dataset

The evaluation scores of the example APS presented in the paper are available in the `merged.csv` file.  
Following the instructions below, you can reproduce this APS dataset from raw data.

### Installing requirements

This project was tested with Python 3.10 on Windows and Linux.  
There are three options to install the requirements.

1. Build a Docker container with the provided `Dockerfile`.
2. Build a Singularity container with the provided definition file `mtl.def`.
3. Install the requirements to your local Python environment with `pip install -r requirements.txt`.

### Reproducibility Instructions

To reproduce the acquisition of the meta-dataset, follow the instructions below.  
This includes downloading, pre-processing, training, and evaluating all algorithms on all datasets.  
An HPC cluster is recommended for this task as this requires a significant amount of computational resources.  
For convenience, the meta-dataset is already included in the repository, so these steps are not necessary to reproduce
only the algorithm selection results.

1. Download the source file from the link in the dataset table and place it in
   the `data_sets/<data_set_name>/source/files/` directory and remove the placeholder file in that directory.
2. Convert the raw data to the processed data.
    1. To run on HPC: run `hpc_convert_raw_to_processed.py`.
    2. To run locally: run `run_convert_raw_to_processed.py --data_set_name <data_set_name>` for each `data_set_name`
       in `hpc_data_set_names.data_set_names`.
3. Convert the processed data to the atomic data.
    1. To run on HPC: run `hpc_convert_processed_to_atomic.py`.
    2. To run locally: run `run_convert_processed_to_atomic.py --data_set_name <data_set_name>` for each `data_set_name`
       in `hpc_data_set_names.data_set_names`.
4. Train all algorithms on all datasets.
    1. To run on HPC: run `hpc_recbole_fit.py`.
    2. To run locally:
       run `run_recbole_fit.py --data_set_name <data_set_name> --algorithm_name <algorithm_name> --algorithm_config <algorithm_config> --fold <fold>`
       for each `data_set_name` in `hpc_data_set_names.data_set_names` for each `algorithm_name`
       in `hpc_data_set_names.algorithm_names` for each `algorithm_config` retrieved by
       calling `recbole_algorithm_config.retrieve_configurations(algorithm_name)` for each `fold` in `[0,1,2,3,4]`.
5. Create predictions with all algorithms on all datasets.
    1. To run on HPC: run `hpc_recbole_predict.py`.
    2. To run locally:
       run `run_recbole_predict.py --data_set_name <data_set_name> --algorithm_name <algorithm_name> --algorithm_config <algorithm_config> --fold <fold>`
       for each `data_set_name` in `hpc_data_set_names.data_set_names` for each `algorithm_name`
       in `hpc_data_set_names.algorithm_names` for each `algorithm_config` retrieved by
       calling `recbole_algorithm_config.retrieve_configurations(algorithm_name)` for each `fold` in `[0,1,2,3,4]`.
6. Evaluate the predictions of all algorithms on all datasets.
    1. To run on HPC: run `hpc_recbole_evaluate.py`.
    2. To run locally:
       run `run_recbole_evaluate.py --data_set_name <data_set_name> --algorithm_name <algorithm_name> --algorithm_config <algorithm_config> --fold <fold>`
       for each `data_set_name` in `hpc_data_set_names.data_set_names` for each `algorithm_name`
       in `hpc_data_set_names.algorithm_names` for each `algorithm_config` retrieved by
       calling `recbole_algorithm_config.retrieve_configurations(algorithm_name)` for each `fold` in `[0,1,2,3,4]`.
7. Run `process_evaluation.py` to aggregate the results of the evaluations of all algorithms on all datasets and create
   the APS dataset in `merged.csv`.

