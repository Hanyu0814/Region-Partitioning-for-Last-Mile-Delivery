# Region-Partitioning-for-Last-Mile-Delivery
## INTRODUCTION:
- Code supplement for the paper "[Provably Good Region Partitioning For On-time Last-mile Delivery](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3915544)" by John Gunnar Carlsson, Sheng Liu, Nooshin Salari, and Han Yu.
- This paper is published on Operations Research.

## KEY DESIGN:
- Optimal Region Partitioning Policy of two measures (Delivery time and service time).
  <img width="657" alt="Screen Shot 2023-08-21 at 10 44 13 AM" src="https://github.com/Hanyu0814/Region-Partitioning-for-Last-Mile-Delivery/assets/42761678/e42bce8f-b108-49f7-b27c-b0c361a4337e">


## REQUIREMENTS:
- Please see requirement.txt
- Run  `pip install -r requirements.txt` to install all required files.

## DATASETS:
- The data and preprocessing code used for the case study are provided in the folder "data". 
- See the "README" file in the folder "data" for a detailed introduction.

## CODE:
Code of all algorithms used in our computational experiments is provided in the folder "code"
- base.py: Helper functions utilized in the algorithm
- sim_par.py: Simulation of different partition policies with varying distributions
- simulation.py: Auxiliary simulation functions for the region partitioning policy
- Two_partition.py: Implementation of the two-partition algorithm
- Three_partition.py: Implementation of the three-partition algorithm


## EXAMPLE USAGE:
- Partition: Run partition_region from sim_par.py;
- For the case study, define measurable functions using the estimated density function, service time, and distance function from "data_preprocessing.ipynb".
- Partition and simulation: Run sim_par_all_uniform from sim_par.py for uniform density function.
