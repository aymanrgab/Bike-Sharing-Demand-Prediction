# Bike Sharing Demand Prediction

This project implements a linear regression model to predict bike sharing demand using the Bike Sharing Dataset from the UCI Machine Learning Repository.

## Project Structure

    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   └── processed      <- The final, canonical data sets for modeling.
    │
    ├── models             <- Trained and serialized models
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment
    │
    └── src                <- Source code for use in this project.
        ├── data           <- Scripts to download or generate data
        │   └── make_dataset.py
        │
        ├── models         <- Scripts to train models and then use trained models to make
        │   │                 predictions
        │   ├── predict_model.py
        │   └── train_model.py
        │
        └── visualization  <- Scripts to create exploratory and results oriented visualizations
            └── visualize.py

## Getting Started

1. Clone this repository.
2. Install the required packages: `pip install -r requirements.txt`
3. Run the data processing script: `python src/data/make_dataset.py`
4. Train the model: `python src/models/train_model.py`
5. Make predictions: `python src/models/predict_model.py`
6. Visualize the results: `python src/visualization/visualize.py`

## License

This project is licensed under the MIT License.
