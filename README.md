# Decision-Focused Probabilistic Forecast Combination

Code supplement to [Decision-Focused Probabilistic Forecast Combination]() (link for preprint soon).

### Abstract

Combining multiple forecasts has long been known to improve forecast quality, as measured by scoring rules in the case of probabilistic forecasting.
However, improved forecast quality does not always translate into better decisions in a downstream problem that utilizes the resultant combined forecast as input.
This work proposes a novel probabilistic forecast combination approach that accounts for the downstream stochastic optimization problem by which the decisions will be made.
We propose a linear pool of probabilistic forecasts where the respective weights are learned by minimizing the expected decision cost of the induced combination,
which we formulate as a nested optimization problem.
Two methods are proposed for its solution:
a gradient-based method that utilizes differential optimization layers and a performance-based weighting method.
For experimental validation, we examine two integral problems associated with renewable energy integration in modern power systems and compare them against well-established combination methods based on linear pooling.
Namely, we examine an electricity market trading problem under stochastic solar production
and a grid scheduling problem under stochastic wind production.
The results illustrate that the proposed decision-focused combination approach leads to lower expected downstream costs while optimizing for forecast quality when estimating linear pool weights does not always translate into better decisions.
Notably, optimizing for a combination of downstream cost and a standard scoring rule consistently leads to better decisions while maintaining high forecast quality.

---

### Code organization

- ```plot_results.py```: generates plots and tables.
- ```main_synthetic_data.py```: runs the synthetic data experiment.
- ```main_solar_trading.py```: runs the trading experiment with stochastic solar production.
- ```main_NEW_wind_grid_scheduling.py```: runs the grid scheduling experiment with stochastic wind production.
- ```torch_layers_functions.py```: includes all implemented pytorch layers.
- ```optimal_transport_functions.py, optimization_functions.py,``` and ```utility_functions```: auxiliary helper functions.
- data: pre-processed and cleaned GEFCom2014 data set.
---

## Citation
Please use the following reference:

```@unpublished{stratigakos:hal-04268454,
  TITLE = {{Decision-Focused Data Pooling for Contextual Stochastic Optimization}},
  AUTHOR = {Stratigakos, Akylas and Morales, Juan Miguel and Pineda, Salvador and Kariniotakis, Georges},
  URL = {https://hal.science/hal-04268454},
  NOTE = {working paper or preprint},
  YEAR = {2023},
  MONTH = Nov,
  KEYWORDS = {Decision support systems contextual stochastic optimization prescriptive analytics data pooling ; Decision support systems ; contextual stochastic optimization ; prescriptive analytics ; data pooling},
  PDF = {https://hal.science/hal-04268454/file/data-pooling-main.pdf},
  HAL_ID = {hal-04268454},
  HAL_VERSION = {v1},
}```
