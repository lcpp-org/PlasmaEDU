## References

[1] He, Yang, et al. Volume-Preserving Algorithms for Charged Particle  Dynamics. Journal of Computational Physics, vol. 281, no. 1, 22 Oct. 2014.

## Summary

This code implements the Boris-Bunemann time integration scheme for charged
particle trajectories in an electromagnetic field. It also implements the
Gh2 algorithm from [1]. The goal is to compare the two schemes.

## Requirements

Python 3 is required

## Setup
Create a new virtual environment and install the dependencies:
```bash
pew new <custom_env_name>
pip install -r requirements.txt
```

## Example

Run source code sanity-check with Gh2 algorithm:
```bash
python src/charged_particle.py gh2
```

Run example script with Gh2 algorithm:
```bash
python scripts/e_cross_b_guiding_center_drift.py
```

