# General 6 DOF manipulator example

The following demonstrates how run the offline pipeline for a general 6 DOF manipulator with 2 % uncertainty in link masses. It 
also demonstrates how to run the controller online with our corridor planning approach.

The collisions geometry for the manipulator is made to over-approximates the wrist joints, i.e., the last three DOFs, reducing the configuration-space to 3 DOF.
The example includes a pretrained 3 DOF nSCDF which uses spheres as obstacle representation.  


## Offline pipeline (optional)
1. A Mosek license is required to run the offline pipeline.
2. Run offline pipline:

`python 1_run_offline_pipline.py`

## Online Corridor Control
1. Install pytorch with CPU version.

`
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cpu
`

3. Run all methods and print results to terminal:

`python 2_run_all.py`

### World space animation
The animation shows the following visualizations:
1) Obstacles (gray spheres)
2) Start configuration (transparent red) 
3) Goal configuration (transparent green) 
4) Configuration along closed loop trajectory (gray)

Run animation of our controller:

`python 3_show_world_space_motion.py`

### Configuration space animation
The animation is visualized in the 3 dimensional configuration space, which are the first 3 DOF of the manipulator. The animation includes the following: 
1) Obstacle region (gray mesh)
2) Start and goal (red and green spheres)
2) Safe corridor (transparent red mesh) 
3) Centerline of corridor (black curve) 
4) Optimized MPC path at time step (blue curve)

Run animation of our controller:

`python 4_show_conf_space_motion.py`
