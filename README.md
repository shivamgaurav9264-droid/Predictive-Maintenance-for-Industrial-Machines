# Predictive Maintenance for Industrial Machines

## Project Overview

Predictive Maintenance is a machine learning project that predicts whether an industrial machine is likely to fail based on sensor readings.

The project contains two machine learning pipelines:

1. Binary Classification
   - Predicts whether the machine will fail.
   - Target: Machine Failure

2. Multi-Class Classification
   - Predicts the specific type of failure.
   - Classes:
     - No Failure
     - Tool Wear Failure (TWF)
     - Heat Dissipation Failure (HDF)
     - Power Failure (PWF)
     - Overstrain Failure (OSF)
     - Random Failure (RNF)

The system is designed to help maintenance teams identify potential machine failures early and perform preventive maintenance.

## Business Objective

The objective is to predict machine failures before they occur so that maintenance can be scheduled proactively instead of reactively.

The binary model provides an early warning, while the multi-class model provides information about the likely failure type.

## Dataset

Dataset:

AI4I 2020 Predictive Maintenance Dataset

Source:

UCI Machine Learning Repository / Kaggle

Dataset contains 10,000 machine records and industrial sensor measurements.

Important features include:

- Air Temperature
- Process Temperature
- Rotational Speed
- Torque
- Tool Wear
- Machine Type

Failure indicators:

- TWF
- HDF
- PWF
- OSF
- RNF

## Machine Learning Pipelines

### Binary Classification

The binary pipeline predicts:

```text
0 = No Failure
1 = Machine Failure