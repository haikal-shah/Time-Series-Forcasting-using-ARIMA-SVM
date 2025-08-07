## Time Series Forecasting using ARIMA + SVM

This web app is built using Streamlit, allowing users to upload time series data with timestamps and forecast future values using a combination of **ARIMA** and **Support Vector Machines (SVM)**. It supports forecasting on an hourly or daily basis and provides visualizations and downloadable results.

---

### Features

* Upload CSV files with datetime and numeric target columns
* Preview and visualize historical data
* Choose forecast horizon (in hours or days)
* Forecast future values using both ARIMA and SVM
* Compare forecasting models
* Download forecast results as a CSV file

---

### Sample Data

You can use any CSV file that includes:

* A **datetime column**
* A **numeric column** representing the target variable to forecast

Example dataset, obtained from [City-Scale Electricity Use Dataset](https://github.com/LBNL-ETA/City-Scale-Electricity-Use-Prediction)

---

## Installation

1. **Clone the repository**

```bash
git clone https://github.com/haikal-shah/Time-Series-Forcasting-using-ARIMA-SVM.git
cd Time-Series-Forcasting-using-ARIMA-SVM
```

2. **Install dependencies**

```bash
pip install -r requirements.txt
```

---

### Run the App

```bash
streamlit run app.py
```

The app will open in your default browser.

---

### Usage Instructions

1. Upload a CSV file with a datetime column and at least one numeric column.
2. Select the appropriate datetime and target variable columns.
3. Choose the forecast unit: **Hours** or **Days**.
4. Set the forecast horizon (e.g., 24 hours or 7 days).
5. View and compare the ARIMA and SVM forecast outputs.
6. Download the forecast results as a CSV file.

---

### Models Used

* **ARIMA (AutoRegressive Integrated Moving Average)**
  A statistical model for analyzing and forecasting time series data.

* **SVM (Support Vector Machine)**
  A machine learning model that uses engineered time-based features like:

  * Ordinal date
  * Hour of day
  * Day of the week

---

### Requirements

All required Python packages are listed in `requirements.txt`:

```
streamlit
pandas
matplotlib
scikit-learn
statsmodels
numpy
```

---

### 📄 License

This project is open-source and available under the [MIT License](LICENSE).

Just say the word!
