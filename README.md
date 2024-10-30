# Decision-Focused Linear Pooling for Probabilistic Forecast Combination

This repository contains the code to recreate the experiments of

```
@unpublished{stratigakos:hal-04593114,
TITLE = {{Decision-Focused Linear Pooling for Probabilistic Forecast Combination}}, AUTHOR = {Stratigakos, Akylas and Pineda, Salvador and Morales, Juan Miguel},  URL = {https://hal.science/hal-04593114}, NOTE = {working paper or preprint}, YEAR = {2024},  MONTH = Sep}
```

which is available [here](https://hal.science/hal-04593114).

### Abstract

Combining multiple forecasts has long been known to improve forecast quality, as measured by scoring rules in the case of probabilistic forecasting.
However, improved forecast quality does not always translate into better decisions in a downstream problem that utilizes the resultant combined forecast as input.
This work proposes a novel probabilistic forecast combination approach that accounts for the downstream stochastic optimization problem by which the decisions will be made.
We propose a linear pool of probabilistic forecasts where the respective weights are learned by minimizing the expected decision cost of the induced combination,
which we formulate as a nested optimization problem.
Two methods are proposed for its solution:
a gradient-based method that utilizes differentiable optimization layers and a performance-based weighting method.
For experimental validation, we examine two integral problems associated with renewable energy integration in modern power systems and compare them against well-established combination methods based on linear pooling.
Namely, we examine an electricity market trading problem under stochastic solar production
and a grid scheduling problem under stochastic wind production.
The results illustrate that the proposed decision-focused combination approach leads to lower expected downstream costs while optimizing for forecast quality when estimating linear pool weights does not always translate into better decisions.
Notably, optimizing for a combination of downstream cost and a standard scoring rule consistently leads to better decisions while maintaining high forecast quality.

---

### Code organization

Running the experiments:
- ```main_synthetic_data.py```: Run the synthetic data experiment.
- ```main_solar_trading.py```: Run the trading experiment with stochastic solar production and create Figure 1.
- ```main_NEW_wind_grid_scheduling.py```: Run the grid scheduling experiment with stochastic wind production and create Figure 3.

In each script, use ```params``` function to set the design experiments.
To store new results, set ```params['save'] = True```.
For the trading case study with stochastic solar production, run ```main_solar_trading.py``` seperately for each target solar plant, as controlled from ```params['target_zone']```.

Other scripts:
- ```plot_results.py```: Generates Tables 1-3 and Figure 2.
- ```torch_layers_functions.py```: Includes all implemented pytorch layers.
- ```utility_functions.py```: Includes auxiliary functions.

Input/ output data and required packages:
- ```\data```: Solar and wind data from the [Global Energy Forecasting competition (GEFCom2014)](https://www.sciencedirect.com/science/article/pii/S0169207016000133?via=ihub#s000140), 
concatenated in a single file.
- ```\data\pglib-opf-cases```: IEEE matpower cases from [Power Grid Lib - Optimal Power Flow](https://github.com/power-grid-lib/pglib-opf) available under a [Creative Commons Attribution License](https://creativecommons.org/licenses/by/4.0/), reproduced for convenience.
- ```\results```: Stores results and trained models for each experiment.
- ```requirements.txt```: Required packages and libraries.
---

### Set-up

This code has been developed using ```Python```, version ```3.9.18```. To install the necessary packages, create a virtual enviroment and run ```pip install -r requirements.txt```.

Contact details: ```a.stratigakos@imperial.ac.uk```.

### Citation
Please use the reference provided above.
