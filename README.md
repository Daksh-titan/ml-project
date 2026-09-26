# MQTF Research Analysis Tool

Streamlit implementation of the Multivariate Quality Trajectory Framework for the fermented millet beverage dataset.

## Source of truth

`MQTF_final_reproducible.ipynb` is the methodological source of truth for the original MQTF pipeline.

`mqtf_core.py` contains the verified implementation and dataset, plus a clearly separated forecasting extension.

## Dataset

6 treatments × 5 storage days × 18 variables, with reported mean and SD values for n=3.

## Reproducibility checks

The verified day-28 regression values are:

STD: 4.352, 4.569, 4.134, 3.916, 4.134, 4.787

FTD: 0.021, 3.004, 3.047, 1.553, 3.015, 1.576

For Monte Carlo feature-selection stability, seed 42 with 1000 simulations gives Acidity 1000/1000 in Domain A and Phytate 1000/1000 in Domain B.

## Forecasting extension

The forecasting tab is explicitly separate from the original MQTF methodology. It uses time-ordered expanding-window backtesting and reports MAE, RMSE, R² and MAPE where appropriate.
