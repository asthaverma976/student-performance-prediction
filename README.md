# Student Performance Prediction System

An end-to-end **Machine Learning** project that predicts student academic performance based on study habits, attendance, sleep patterns, and other factors. Built with **Flask**, **Scikit-learn**, and an optional **MongoDB** backend.

---

## Features

- **Synthetic Data Generation** - Generates 1,000 realistic student records with correlated features
- **Exploratory Data Analysis** - Correlation heatmaps, distribution plots, feature importance charts
- **Multi-Model Training** - Compares Linear Regression, Random Forest, and Decision Tree; auto-selects the best
- **Web Interface** - Premium dark-themed UI with glassmorphism, gradient backgrounds, and smooth animations
- **Real-time Predictions** - Enter student data and get instant performance predictions with confidence levels
- **MongoDB Integration** - Optional database storage for student records and prediction history

---

## Tech Stack

| Layer         | Technology                            |
|---------------|---------------------------------------|
| **Backend**   | Python 3, Flask                       |
| **ML Models** | Scikit-learn (Linear Regression, Random Forest, Decision Tree) |
| **Data**      | Pandas, NumPy                         |
| **Viz**       | Matplotlib, Seaborn                   |
| **Database**  | MongoDB (optional via PyMongo)        |
| **Frontend**  | HTML5, CSS3 (Glassmorphism Dark UI)   |

---

## Project Structure

```
student-performance-prediction/
├── app.py                      # Flask web application
├── model/
│   ├── train.py                # Model training & comparison
│   ├── predict.py              # Prediction logic
│   ├── best_model.pkl          # (generated) Best trained model
│   └── scaler.pkl              # (generated) Feature scaler
├── data/
│   ├── generate_data.py        # Synthetic data generator
│   └── students.csv            # (generated) Dataset
├── analysis/
│   ├── eda.py                  # Exploratory Data Analysis
│   └── plots/                  # (generated) EDA visualizations
├── database/
│   └── mongo_handler.py        # MongoDB handler (optional)
├── templates/
│   ├── index.html              # Input form
│   └── result.html             # Prediction result
├── static/
│   └── style.css               # Premium dark theme styling
├── requirements.txt
└── README.md
```

---

## Installation & Setup

### 1. Navigate to the project

```bash
cd student-performance-prediction
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. (Optional) Start MongoDB

If you have MongoDB installed, start the `mongod` service. The app works without it too - all MongoDB operations fail gracefully.

---

## How to Run

Execute these commands **in order**:

### Step 1 - Generate the dataset

```bash
python data/generate_data.py
```

Generates `data/students.csv` with 1,000 synthetic student records.

### Step 2 - Train the ML model

```bash
python model/train.py
```

Trains 3 models, compares them, and saves the best model to `model/best_model.pkl`.

### Step 3 - (Optional) Run EDA

```bash
python analysis/eda.py
```

Generates correlation heatmaps, distribution plots, and feature importance charts in `analysis/plots/`.

### Step 4 - Launch the web app

```bash
python app.py
```

Open **http://127.0.0.1:5000** in your browser.

---

## Model Performance

| Model              | R²     | MAE   | MSE    |
|---------------------|--------|-------|--------|
| Linear Regression   | ~0.88  | ~4.1  | ~27.5  |
| Random Forest       | ~0.94  | ~2.8  | ~13.2  |
| Decision Tree       | ~0.89  | ~3.6  | ~24.8  |

*Random Forest is typically selected as the best model.*

---

## Input Features

| Feature                      | Range   | Description                       |
|------------------------------|---------|-----------------------------------|
| `study_hours`                | 0-12    | Daily study hours                 |
| `attendance_percentage`      | 0-100   | Class attendance rate             |
| `previous_scores`            | 0-100   | Previous academic scores          |
| `sleep_hours`                | 4-10    | Daily sleep hours                 |
| `extracurricular_activities` | 0 / 1   | Participation in activities       |
| `parent_education_level`     | 0-4     | Highest parental education level  |

---

## How It Works

1. **Data Generation** - Synthetic student data is generated with realistic correlations between features
2. **Preprocessing** - Features are scaled using StandardScaler for optimal model performance
3. **Model Training** - Three regression models are trained and compared on R², MAE, and MSE metrics
4. **Best Model Selection** - The model with the highest R² score is automatically saved
5. **Web Prediction** - Flask serves a form where users input student data and receive instant predictions

---

## MongoDB (Optional)

The application works **without MongoDB**. If you want to enable database storage:

1. Install and start MongoDB locally (`mongodb://localhost:27017/`)
2. The app will automatically detect and connect to MongoDB
3. Student records and predictions will be stored in the `student_performance_db` database

---

## License

This project is open source and available under the [MIT License](LICENSE).
