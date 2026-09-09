# Student Performance Predictor

A small ML project that predicts whether a student will **Pass** or **Fail**
based on their study hours, attendance, previous exam score, and whether
they take part in extracurricular activities.

## How it works

1. `generate_data.py` creates a fake dataset of 300 students (`data/students.csv`).
   There's no real dataset here — the pass/fail label is generated from a simple
   weighted formula (study hours + attendance + previous score matter most)
   plus some random noise, so the data isn't perfectly predictable.
2. `train_model.py` trains a **Logistic Regression** model on that data and
   saves it as `model.pkl`.
3. `app.py` is a Streamlit web app where you enter a student's details and
   get a Pass/Fail prediction with a probability.

## Project structure

```
student-performance-predictor/
├── data/
│   └── students.csv       # generated dataset
├── generate_data.py        # creates the dataset
├── train_model.py          # trains and saves the model
├── model.pkl                # saved model (created after running train_model.py)
├── app.py                  # Streamlit app
└── requirements.txt
```

## How to run

```bash
pip install -r requirements.txt

# 1. generate the dataset
python generate_data.py

# 2. train the model
python train_model.py

# 3. run the app
streamlit run app.py
```

Then open the local URL Streamlit prints (usually http://localhost:8501).

## Why Logistic Regression?

The target is a simple yes/no outcome (Pass or Fail), and Logistic Regression
is a standard, easy-to-explain algorithm for binary classification — it
outputs a probability (via the sigmoid function) rather than just a hard
label, which is why the app can show a "% chance of passing" instead of
just Pass/Fail. On the held-out test set it gets about 83% accuracy.

## Possible improvements

- Try other models (Decision Tree, Random Forest) and compare accuracy.
- Use a real dataset instead of a generated one.
- Add more features (e.g. sleep hours, assignment scores).
