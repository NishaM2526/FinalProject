# COVID-19 X-Ray Detection

Lightweight Streamlit app that classifies chest X-Ray images into `Covid`, `Normal`, or `Viral Pneumonia` using a pre-trained Keras model.

## Project structure

- App/: Streamlit application entry (`app.py`).
- Code/: Notebooks and development artifacts.
- Data/: Dataset folders used for training/validation (not required to run the demo).
- Model/: Trained Keras model files (app expects `vgg16_covid_model.keras`).

## Requirements

- Python 3.8+
- Packages: `streamlit`, `tensorflow`, `pillow`, `numpy`, `pandas`

You can install the main dependencies with:

```
pip install streamlit tensorflow pillow numpy pandas
```

If you maintain a `requirements.txt` you can run:

```
pip install -r requirements.txt
```

## Tech Stack

- **Frontend / App:** Streamlit
- **Model & DL framework:** TensorFlow / Keras
- **Python packages:** `numpy`, `pandas`, `pillow`
- **Model format:** Keras `.keras` model file (TensorFlow SavedModel/HDF5 compatible exports)
- **Development:** Jupyter Notebooks in `Code/` for training and experiments

## Run the app

From the project root (`Covid19Detection`), run:

```
streamlit run App/app.py
```

The app loads the model from `Model/vgg16_covid_model.keras`. Ensure that file exists before running — the app will attempt to load it from the `Model` folder relative to the project root.

## Usage

- Open the Streamlit UI in your browser when the server starts.
- Upload a chest X-ray image (`jpg`, `jpeg`, or `png`).
- The app will display the predicted class and confidence, plus a bar chart of class probabilities.

## Notes

- The repo includes training notebooks in `Code/` used to create and evaluate models. The demo app is intended for inference only — do not use it as a medical diagnostic tool.
- If you want to retrain or evaluate models, inspect the notebooks in `Code/` and ensure you have appropriate GPU drivers for TensorFlow if required.

## License & attribution

This project is for educational purposes. See the repository owner for license details.

## Author

- Nisha M

---
Created for the COVID-19 X-Ray Detection demo.
