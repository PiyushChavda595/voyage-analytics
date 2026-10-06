# ✈️ Voyage Analytics

### AI-Powered Travel Recommendation & Prediction System

Voyage Analytics is an end-to-end travel analytics platform that combines **Machine Learning, FastAPI, Streamlit, MLflow, Docker, and Kubernetes** to provide travel-related predictions and recommendations.

The system provides three main capabilities:

- ✈️ Flight Price Prediction
- 🏨 Travel and Hotel Recommendations
- 👤 Travel Behavior-Based Gender Classification

The project demonstrates the complete workflow from data preprocessing and model training to API development, frontend integration, containerization, experiment tracking, and deployment.

---

## 🌐 Live Demo

### Frontend

https://voyage-analytics-mgng.onrender.com

### Backend API

https://voyage-backend-09jx.onrender.com

### GitHub Repository

https://github.com/PiyushChavda595/voyage-analytics

> **Note:** The backend is deployed on Render and may take some time to respond after a period of inactivity.

---

## 📌 Features

### ✈️ Flight Price Prediction

Predicts estimated flight prices using a trained **XGBoost regression model**.

### 🏨 Travel & Hotel Recommendation

Recommends destinations and associated hotels based on historical travel behavior using **cosine similarity**.

### 👤 Gender Classification

Predicts a gender class based on available travel behavior features.

### 📊 Interactive Web Interface

The Streamlit frontend provides an interactive interface for accessing the prediction and recommendation features.

### ⚡ REST API

FastAPI provides REST endpoints for:

- Flight price prediction
- Gender classification
- Travel recommendations

### 🧪 MLflow Experiment Tracking

MLflow is used to track model experiments, parameters, and evaluation metrics such as **R² and RMSE**.

### 🐳 Docker

The frontend and backend are containerized separately and can be run together using Docker Compose.

### ☸️ Kubernetes

The application can also be deployed locally using **Kubernetes and Minikube**.

---

## 🏗️ System Architecture

```text
                    ┌─────────────────────┐
                    │        User         │
                    │     Web Browser     │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Streamlit Frontend  │
                    │      Port 8501      │
                    └──────────┬──────────┘
                               │
                         HTTP Requests
                               │
                               ▼
                    ┌─────────────────────┐
                    │   FastAPI Backend   │
                    │      Port 8000      │
                    └──────────┬──────────┘
                               │
              ┌────────────────┼────────────────┐
              │                │                │
              ▼                ▼                ▼
       ┌────────────┐   ┌─────────────┐  ┌──────────────┐
       │  XGBoost   │   │ Classifier  │  │ Recommendation│
       │ Regression │   │    Model    │  │    Engine     │
       └────────────┘   └─────────────┘  └──────────────┘
              │                │                │
              └────────────────┼────────────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   Travel Datasets   │
                    │ Flights / Hotels /  │
                    │      Users         │
                    └─────────────────────┘
```

---

## 🔄 Application Workflow

```text
Travel Dataset
      ↓
Data Cleaning
      ↓
Feature Engineering
      ↓
Model Training
      ↓
Model Evaluation
      ↓
MLflow Experiment Tracking
      ↓
Model Serialization
      ↓
FastAPI Backend
      ↓
REST API
      ↓
Streamlit Frontend
      ↓
Docker
      ↓
Kubernetes / Render
      ↓
End User
```

---

## 🤖 Machine Learning

### 1. Flight Price Prediction

**Problem Type:** Regression

**Algorithm:** XGBoost Regressor

The flight price prediction model uses historical travel data to estimate flight prices based on relevant features.

#### Training Process

```text
Processed Travel Data
        ↓
Feature Selection
        ↓
Train/Test Split
        ↓
XGBoost Training
        ↓
Prediction
        ↓
R² + RMSE Evaluation
        ↓
MLflow Tracking
        ↓
Model Serialization
```

The trained model is serialized using `joblib` and loaded by the FastAPI backend during inference.

---

### 2. Gender Classification

**Problem Type:** Classification

The project includes a classification model that predicts a gender class from available travel behavior features.

The model is exposed through:

```text
POST /predict-gender
```

The classification module is primarily included as part of the project's end-to-end machine learning pipeline.

---

### 3. Travel Recommendation Engine

The recommendation engine uses **cosine similarity** to identify users with similar travel patterns.

```text
User
 ↓
Travel History
 ↓
Destination Frequency
 ↓
User-Destination Representation
 ↓
Cosine Similarity
 ↓
Similar Users
 ↓
Destination Recommendations
 ↓
Hotel Recommendations
```

For a known user, the system identifies similar travel patterns and recommends destinations. Associated hotel information is then retrieved from the available dataset.

If a user is not available in the similarity data, the system can fall back to popular destinations.

---

## 🖥️ Frontend

The frontend is built using **Streamlit**.

### Responsibilities

- Collect user inputs
- Provide interactive controls
- Send requests to the FastAPI backend
- Display prediction results
- Display travel recommendations
- Provide a simple interface for the ML functionality

### Frontend Port

```text
8501
```

### Local URL

```text
http://localhost:8501
```

---

## ⚙️ Backend

The backend is built using **FastAPI**.

### Responsibilities

- Load trained machine learning models
- Load travel datasets
- Prepare input data
- Perform model inference
- Generate travel recommendations
- Return JSON responses to the frontend

### Backend Port

```text
8000
```

### Local URL

```text
http://localhost:8000
```

### FastAPI Documentation

```text
http://localhost:8000/docs
```

---

## 🔌 API Endpoints

### `GET /`

Checks whether the API is running.

**Example:**

```http
GET /
```

**Response:**

```json
{
  "message": "Voyage Analytics API Running"
}
```

---

### `POST /predict-price`

Predicts the estimated flight price.

**Example request:**

```json
{
  "feature_1": 100,
  "feature_2": 200,
  "feature_3": 50
}
```

> The actual input features depend on the feature set used by the trained model.

**Example response:**

```json
{
  "predicted_price": 1234.56
}
```

---

### `POST /predict-gender`

Predicts the gender class using the trained classification model.

**Example response:**

```json
{
  "predicted_gender": 1
}
```

---

### `GET /recommend-trip`

Generates travel and hotel recommendations for a user.

**Example:**

```http
GET /recommend-trip?user_id=10
```

The response contains recommended destinations and associated hotel information from the available datasets.

---

## 📊 Data Processing

The project processes travel datasets before they are used by the machine learning models.

The preprocessing workflow includes:

- Handling missing values where required
- Encoding categorical variables
- Selecting relevant features
- Removing leakage-related columns
- Preparing numerical features
- Creating model-ready datasets

---

## 🧪 MLflow

MLflow is used to track experiments for the flight price prediction model.

Different XGBoost configurations can be evaluated and tracked using parameters such as:

```text
n_estimators
max_depth
learning_rate
```

The following evaluation metrics are tracked:

```text
R²
RMSE
```

This makes it easier to compare different model configurations and identify better-performing experiments.

---

## 🐳 Docker

The project contains Docker configurations for both the frontend and backend.

The root `docker-compose.yml` runs:

```text
Backend  →  Port 8000
Frontend →  Port 8501
```

### Run the complete application

From the project root:

```bash
docker compose up --build
```

### Stop the application

```bash
docker compose down
```

### Rebuild the application

```bash
docker compose up --build
```

---

## ☸️ Kubernetes

The project also demonstrates local Kubernetes deployment using **Minikube**.

### Deployment Flow

```text
Source Code
     ↓
Docker Image
     ↓
Kubernetes
     ↓
Minikube
     ↓
Pods / Services
     ↓
Application
```

Kubernetes is used to demonstrate container orchestration and deployment concepts in a local environment.

---

## ☁️ Deployment

### Render

The project is publicly deployed using Render.

```text
Frontend
   ↓
Render
   ↓
Streamlit Application
```

```text
Backend
   ↓
Render
   ↓
FastAPI Application
```

### Live Deployment

**Frontend**

https://voyage-analytics-mgng.onrender.com

**Backend**

https://voyage-backend-09jx.onrender.com

> The backend may sleep after being inactive, so the first request after inactivity can take longer.

---

## 📁 Project Structure

```text
voyage-analytics/
│
├── .github/
│   └── workflows/
│
├── backend/
│   ├── data/
│   │   ├── flights.csv
│   │   ├── hotels.csv
│   │   └── ...
│   │
│   ├── models/
│   │   ├── final_clean_model.pkl
│   │   ├── features.pkl
│   │   ├── gender_model_final.pkl
│   │   └── gender_features.pkl
│   │
│   ├── main.py
│   ├── requirements.txt
│   └── Dockerfile
│
├── frontend/
│   ├── ...
│   └── Dockerfile
│
├── train_mlflow.py
├── docker-compose.yml
├── .gitignore
└── README.md
```

---

## 🛠️ Technology Stack

| Category | Technology |
|---|---|
| Programming Language | Python |
| Frontend | Streamlit |
| Backend | FastAPI |
| API Server | Uvicorn |
| ML Regression | XGBoost |
| ML Classification | Scikit-learn |
| Data Processing | Pandas, NumPy |
| Recommendation | Cosine Similarity |
| Model Serialization | Joblib |
| Experiment Tracking | MLflow |
| Containerization | Docker |
| Container Orchestration | Kubernetes |
| Local Kubernetes | Minikube |
| Deployment | Render |
| Version Control | Git / GitHub |

---

## ⚙️ Installation

### Prerequisites

Make sure the following are installed:

- Python 3.10+
- Git
- Docker
- Docker Compose
- Minikube, if using Kubernetes
- kubectl, if using Kubernetes

---

## 📥 Clone the Repository

```bash
git clone https://github.com/PiyushChavda595/voyage-analytics.git
```

```bash
cd voyage-analytics
```

---

## 🐍 Backend Setup

Move into the backend directory:

```bash
cd backend
```

### Create a Virtual Environment

#### Windows

```powershell
python -m venv venv
```

Activate the environment:

```powershell
.\venv\Scripts\activate
```

#### Linux / macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

The backend uses libraries including:

```text
FastAPI
Uvicorn
Pandas
NumPy
Scikit-learn
XGBoost
Joblib
```

---

## ▶️ Run the Backend

From the `backend` directory:

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at:

```text
http://localhost:8000
```

Interactive API documentation:

```text
http://localhost:8000/docs
```

---

## 🖥️ Run the Frontend

The frontend uses Streamlit.

From the frontend directory, install the required dependencies and run the Streamlit entry file:

```bash
streamlit run <frontend-entry-file>.py
```

The frontend will normally be available at:

```text
http://localhost:8501
```

The Streamlit application communicates with the FastAPI backend through HTTP requests.

---

## 🐳 Run with Docker Compose

From the project root:

```bash
docker compose up --build
```

Once the containers are running:

**Backend**

```text
http://localhost:8000
```

**Frontend**

```text
http://localhost:8501
```

To stop the containers:

```bash
docker compose down
```

---

## 🧪 Model Training

The project includes:

```text
train_mlflow.py
```

The training workflow is:

```text
Processed Travel Data
        ↓
Feature Selection
        ↓
Train/Test Split
        ↓
XGBoost Training
        ↓
Model Evaluation
        ↓
MLflow Logging
        ↓
Model Saving
```

Run the training script with:

```bash
python train_mlflow.py
```

The trained model and feature information are saved using `joblib`.

---

## 📡 API Testing

FastAPI provides an interactive Swagger UI.

Open:

```text
http://localhost:8000/docs
```

Available endpoints:

```text
GET  /
POST /predict-price
POST /predict-gender
GET  /recommend-trip
```

---

## 📈 Performance & Observations

### Flight Price Prediction

The XGBoost regression model demonstrated good predictive performance during experimentation based on the evaluation metrics tracked through MLflow.

### Gender Classification

The classification model showed relatively limited predictive performance during experimentation and is intended primarily as a demonstration component of the project pipeline.

### Recommendation Engine

The recommendation system:

- Uses cosine similarity
- Identifies similar user travel patterns
- Generates destination recommendations
- Connects destinations with available hotel information
- Does not require a separate supervised training process

---

## ⚠️ Limitations

### Historical Data

Predictions and recommendations are based on the available datasets and should not be treated as live travel information.

### Flight Prices

The predicted price is an ML estimate, not a live airline quotation.

Actual prices can change based on:

- Demand
- Availability
- Booking time
- Airline pricing
- Seasonal changes
- Other real-world factors

### Hotel Recommendations

Hotel recommendations are based on the available dataset and do not represent:

- Live hotel inventory
- Live room availability
- Live booking prices

### Gender Classification

The gender classification component has limited predictive performance and should not be considered a reliable real-world demographic inference system.

### Cloud Deployment

The Render deployment may experience a slower first request after inactivity because of the hosting environment.

### Kubernetes

The Kubernetes deployment is demonstrated locally using Minikube rather than a production cloud Kubernetes cluster.

---

## 🔮 Future Scope

Potential improvements include:

### Machine Learning

- Improved feature engineering
- Hyperparameter optimization
- Additional regression algorithms
- Model explainability
- Improved recommendation techniques

### Real-Time Travel Data

Integrate real-time APIs for:

- Flight prices
- Flight availability
- Hotel availability
- Hotel prices
- Booking information

### Cloud Deployment

The application could be deployed to cloud Kubernetes platforms such as:

- AWS EKS
- Google Kubernetes Engine
- Azure Kubernetes Service

### CI/CD

A complete CI/CD pipeline could include:

- GitHub Actions
- Automated testing
- Docker image builds
- Automated deployment

### Monitoring

Future monitoring can include:

```text
Prometheus
     +
Grafana
```

for monitoring application health, API performance, resource utilization, and model-serving performance.

---

## 👥 Team

- **Piyush Chavda**
- **Dipanshu Bagde**
- **Shinde Akash**
- **Jayshree Pawar**

---

## 📅 Project Information

**Project:** Voyage Analytics  
**Type:** AI-Powered Travel Recommendation & Prediction System  
**Date:** April 21, 2026

---

## 📄 License

This project is intended for educational, academic, and demonstration purposes.

Please check the repository configuration and the licenses of the datasets and third-party libraries before redistributing or reusing the project.

---

## 🔗 Project Links

- **GitHub:** https://github.com/PiyushChavda595/voyage-analytics
- **Live Frontend:** https://voyage-analytics-mgng.onrender.com
- **Live Backend:** https://voyage-backend-09jx.onrender.com
