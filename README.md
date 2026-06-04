# Robust Convex Model Predictive Control with collision avoidance guarantees for robot manipulators

<p align="center">
<img src="demo.gif"/>
</p>

This is the project repository for the paper:

https://arxiv.org/abs/2508.21677


### Overview of repository:
```bash
.
├── controllers
├── corridor_simulators
├── examples
│   ├── FER
│   ├── planar_two_dof
│   └── six_dof
└── path_planner
```

### Requirements:
- Mosek license (optional)
- Python 3.10

### Installation and setup:
1. Install python packages:

`pip install -r requirements.txt`

2. Add project folder to python path

`export PYTHONPATH="$(pwd):$PYTHONPATH`

### Optional 
1. To be able to run the LMI calculation steps, a Mosek license is required.
2. Install cvxpy to support Mosek:

`pip install cvxpy[MOSEK]`


### Examples:
We provide the following examples to get started:

- [Planar 2 DOF manipulator](./examples/planar_two_dof)
- [General 6 DOF manipulator](./examples/six_dof)
- [Data driven computation of model-error constants with a Franka Emika Robot (FER)](./examples/FER)
