# Machine Learning vs. LR-based Nomogram for ORN Prediction After Fibula Flap Reconstruction in Primary Oral Cancer Patients

## Description
This project implements various Machine Learning (ML) models and a logistic regression-based nomogram to predict post-operative occurance of osteoradionecrosis (ORN) in head and neck cancer patients receiving fibula flap reconstruction after tumor excision and segmental mandibulectomy.

## Project layout
- notebooks/: end-to-end machine learning workflows (data cleaning, EDA, preprocessing, tuning, evaluation, and figure/table generation)

- src/: reusable functions shared across notebooks, including a global seed/random state

- data/: local data storage (not tracked; raw data not available for public use at this time)

- results/: generated figures and tables created by notebooks/scripts
  - Not all figures/tables are featured in the correspoding manuscript
 
- models/: trained models available for use

## Installation

### Prerequisites
Ensure you have [Anaconda](https://www.anaconda.com/products/distribution) installed on your system.

### Steps
1. Clone this repository with HTTPS or SSH:

- Using HTTPS:
```
git clone https://github.com/AnthonyMatarr/ML-ORN-Pred.git
```
- Using SSH:
```
git git@github.com:AnthonyMatarr/ML-ORN-Pred.git
```
2. Navigate to project directory
```
cd ML-ORN-Pred
```
3. Create a new virtual environment:
```
conda env create -f environment.yml
```
4. (Optional) If you encounter issues with pip-installed packages, install them manually:
```
pip something
```

## Usage
1. Activate the environment:
```
conda activate orn_env
```
- Note: Due to OS/architecture differences and solver choices, minor numerical deviations from the manuscript may occur
2. Adjust file paths
- Paths in notebooks assume local folder layout
3. Run notebooks
  - Notebooks are numbered by stage, but assuming necessary data is available, can be run on their own
 
## License: MIT
- Code licensed under MIT
- No data are included


