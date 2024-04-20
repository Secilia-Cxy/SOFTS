# SOFTS: Efficient Multivariate Time Series Forecasting with Series-Core Fusion

The code repository for "SOFTS: Efficient Multivariate Time Series Forecasting with Series-Core Fusion" in PyTorch.

## Abstract

Multivariate time series forecasting plays a crucial role in various fields such as finance, traffic management, energy, and healthcare.  Recent studies have highlighted the advantages of channel independence to resist distribution drift but neglect channel correlations, limiting further enhancements. Several methods utilize mechanisms like attention or mixer to address this by capturing channel correlations, but they either introduce excessive complexity or fail to deliver state-of-the-art results, particularly with a large number of channels.
Addressing this gap, this paper presents an efficient MLP-based model, the Series-cOre Fused Time Series Forecaster (SOFTS), which incorporates a novel STar Aggregate-Dispatch (STAD) module. Unlike traditional approaches that manage channel interactions through distributed structures, STAD employs a centralized strategy. It aggregates all series to form a global core representation, which is then dispatched and fused with individual series representations to facilitate channel interactions effectively.
SOFTS not only achieves superior performance over existing state-of-the-art methods with only linear complexity but also demonstrates the broad applicability of the STAD module across different forecasting models. 

#### Main Structure

![structure](imgs\structure.png)

#### Star Aggregate Dispatch Module (STAD)

![STAD](imgs\STAD.png)

#### Performance Comparison

![performance](imgs\performance.png)

#### Efficiency Comparison

![efficiency](imgs\efficiency.png)

## Prerequisites

scikit-learn==1.2.2

numpy==1.22.4

pandas==1.2.4

torch==1.10.0+cu111

## Datasets

We refer to [this repository](https://github.com/thuml/Time-Series-Library) for downloading datasets.

## Scripts

To reproduce the main results in Table 2, run the script files under folder `scripts/long_term_forecast`.

For example, to reproduce the results of SOFTS on ETTm1 dataset, run the following command:

```sh scripts/long_term_forecast/ETTm1.sh```

## Acknowledgement

We appreciate the following github repos a lot for their valuable code base or datasets:

https://github.com/zhouhaoyi/Informer2020

https://github.com/thuml/Autoformer

https://github.com/zhouhaoyi/ETDataset

https://github.com/laiguokun/multivariate-time-series-data

https://github.com/thuml/Time-Series-Library