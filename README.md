# Age Estimation from Speech

## 📌 Project Overview

This project aims to estimate the age of a speaker based on their vocal characteristics using machine learning techniques. The approach involves feature extraction from speech signals and building a regression model to predict the speaker’s age.

## 📊 Dataset

- **Total samples:** 3,624
- **Train/Test split:** 2,933 training samples, 691 testing samples
- **Features extracted:**
  - Acoustic features: pitch (mean, max, min), jitter, shimmer, energy, zero-crossing rate, spectral centroid
  - Linguistic metadata: gender, ethnicity, speaking rate (tempo), number of pauses, silence duration
  - Speech signal analysis: MFCCs, chroma features

## 🎯 Task

- The objective is to create a **regression pipeline** for predicting a speaker's age from extracted speech features.

## 📏 Evaluation Metric

- Root Mean Square Error (**RMSE**) is used to measure the performance of the model:

  \(RMSE = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}\)

## 🔧 Methodology & Models

### 1️⃣ **Data Preprocessing**

- Noise cancellation applied to enhance speech clarity
- Log transformation on age values to handle outliers
- Feature normalization using `StandardScaler`

### 2️⃣ **Feature Selection**

- Applied **Random Forest feature importance** to select 52 relevant features

### 3️⃣ **Regression Models Evaluated**

| Model                | Performance        |
| -------------------- | ------------------ |
| Decision Tree        | Poor               |
| Random Forest        | Below expectation  |
| Linear Regression    | Moderate           |
| Gradient Boosting    | High performance   |
| **Ridge Regression** | ✅ Best Performance |

- **Final Model:** Ridge Regression with hyperparameter tuning (`alpha=100`, `solver=SAG`, `tol=0.001`)
- Hyperparameter tuning performed using **GridSearchCV** with **K-Fold Cross-Validation**

## 📈 Final Evaluation

- Ridge Regression achieved the best **RMSE score** on the test set.
- Pipeline includes:
  - Noise reduction
  - Feature normalization
  - Ridge model prediction

## 🚀 Installation & Usage

### 🔹 Requirements

Ensure you have the required dependencies installed. Run:

```bash
pip install -r requirements.txt
```

### 🔹 Running the Project

To run the age prediction pipeline, use the following command:

```bash
python src/main.py
```

For Jupyter Notebook:

```bash
jupyter notebook
```

Then open `speech-based-age-estimation.ipynb`.

## 📂 Project Structure

```
age-estimation-speech/
│── data/                     
│── notebooks/                  # Jupyter Notebook files
│   ├── speech-based-age-estimation.ipynb
│── README.md                   # Project description
│── requirements.txt             # List of dependencies

```

## 📜 References

- Scikit-learn: Pedregosa et al., "Machine Learning in Python," JMLR, 2011.
- Librosa: McFee et al., "Audio and music signal analysis in Python," SciPy, 2015.
- Jurafsky & Martin, "Speech and Language Processing," Pearson, 2009.


---

✅ **GitHub Repository:** [https://github.com/mohammad-rahbari/speech-based-age-estimation]

