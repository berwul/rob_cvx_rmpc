# Planar 2 DOF manipulator example

The following demonstrates how to run the offline pipeline for a planar 2 DOF manipulator with 20 % uncertainty in link masses.

## Offline pipeline (optional)
1. A Mosek license is required to run the offline pipeline.
2. Run offline pipline:

`python 1_run_offline_pipline.py`

3. List offline results 

`ls data/dof_2_ef_0.20`

## Online Corridor Control
1. Run all methods:

`python 2_run_all.py`

2. Print results:

`python 3_print_results.py`

3. Run animation of selected method:

`python 4_run_mpc_animation.py --method {method_name}`

where method_name is one of {"nom_star", "rt", "ft"}, where "ft" is our method.
