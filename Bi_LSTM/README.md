# My Project

## Overview
This project is designed to train and predict actions using a Bi-LSTM model with MediaPipe for gesture recognition.

## Setup
1. Clone the repository.
2. Install the required dependencies using:
    ```
    pip install -r requirements.txt
    ```
3. Ensure you have the necessary datasets and pre-trained models placed in the appropriate directories.

## Usage
- To train the model:
    ```
    python scripts/train.py
    ```
- To predict actions:
    ```
    python scripts/predict.py
    ```
- Alternatively, you can use the main script to select the mode:
    ```
    python scripts/main.py
    ```

## Files
- `main.py`: Entry point to run the training or prediction.
- `train.py`: Script to train the model.
- `predict.py`: Script to predict actions using the trained model.
- `utils.py`: Contains utility functions used across the project.


[Watch the video](Arabic_sign_language_traslator.mp4)