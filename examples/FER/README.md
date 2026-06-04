# Franka Emika Robot (FER) example

The purpose of the following example is to show how a controller can be computed from measurement data. We use the data provided from the following repository:

https://git-ce.rwth-aachen.de/jan-niklas-schneider/LSTM-based_inverse_dynamics_learning_for_franka_emika_panda

The data in the link above provides measurements from a FER 7 DOF robot. Download the repo from the above link. Uncompress and copy to the current directory (the example FER directory)

`cp -r {/path/to/dowloaded/dir} {/path/to/FER_example}/LSTM-based_inverse_dynamics_learning_for_franka_emika_panda-main-data`

## Steps:

1. Run the data generation script:

`python 1_compile_data_set.py`

2. Compute controllers:

`python 2_compute_controller.py`

3. Pick controllers:

`python 3_pick_controllers.py`

4. Run all methods in randomized configuration space corridors:

`python 4_run_in_corridor.py`

5. Print the results from the runs to console:

`python 5_show_runs_result.py`
