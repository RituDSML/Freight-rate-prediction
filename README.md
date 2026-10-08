# Freight Rate Prediction - Machine Learning Assessment

This repository contains the baseline machine learning solution and documentation for the Freight Rate Prediction assessment.

## Repository Contents

* explore.py : Exploratory data analysis and data inspection scripts.
* score.py`   : Scoring script used to evaluate the model and generate the evaluation chart (`candidate_december.png`).
* validation_predictions.csv : The output validation predictions file containing `load_id` and `predicted_rate`.
* data/ : Directory containing the training, test, validation datasets and december-chart-inputs.csv
* scorer_results/ : Contains generated visual evaluation outputs (including `candidate_december.png`).
* requirements.txt : Python package dependencies.

## Prerequisites & Dependencies
To install the required dependencies, ensure you have Python installed, activate your virtual environment, and run.
```bash
pip install -r requirements.txt
