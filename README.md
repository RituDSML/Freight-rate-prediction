# Freight Rate Prediction - Machine Learning Project

This repository contains the baseline machine learning solution and documentation for the Freight Rate Prediction challenge.

## Repository Contents

* explore.py : Handles exploratory data analysis, train/test splitting, and model training workflows.
* score.py`   : Scoring script used to evaluate the model and generate the evaluation chart (`candidate_december.png`).
* validation_predictions.csv : The output validation predictions file containing `load_id` and `predicted_rate`.
* data/ : Directory containing the training, test, validation datasets and december-chart-inputs.csv
* scorer_results/ : Contains generated visual evaluation outputs (including `candidate_december.png`).
* requirements.txt : Python package dependencies.

## Prerequisites & Dependencies
To install the required dependencies, ensure you have Python installed, activate your virtual environment, and run.
```bash
pip install -r requirements.txt
```
## Run Instructions
* Run Training & Exploration: Execute the exploratory and training script to process data, perform the train/test split, and prepare the model:
python explore.py

* Generate Predictions & Score: Run the scoring script to validate the model, generate predictions (validation_predictions.csv), and output the fixed December performance chart:
python score.py
