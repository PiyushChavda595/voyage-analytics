from fastapi import FastAPI
import pandas as pd
import joblib

app = FastAPI(title="Voyage Analytics API")

# ==============================
# LOAD MODELS (SAFE LOAD)
# ==============================
try:
    reg_model = joblib.load("models/final_clean_model.pkl")
    reg_features = joblib.load("models/features.pkl")
except:
    reg_model = None
    reg_features = []

try:
    gender_model = joblib.load("models/gender_model_final.pkl")
    gender_features = joblib.load("models/gender_features.pkl")
except:
    gender_model = None
    gender_features = []

# ==============================
# LOAD DATA (SAFE)
# ==============================
try:
    flights = pd.read_csv("data/flights.csv")
    hotels = pd.read_csv("data/hotels.csv")
except:
    flights = pd.DataFrame()
    hotels = pd.DataFrame()

# ==============================
# PRECOMPUTE (SAFE)
# ==============================
if not flights.empty:
    popular_routes = flights.groupby(['from', 'to']).size().reset_index(name='count')
    popular_routes = popular_routes.sort_values(by='count', ascending=False)

    user_route = flights.groupby(['userCode', 'to']).size().unstack(fill_value=0)

    from sklearn.metrics.pairwise import cosine_similarity
    similarity = cosine_similarity(user_route)

    similarity_df = pd.DataFrame(
        similarity,
        index=user_route.index,
        columns=user_route.index
    )
else:
    popular_routes = pd.DataFrame()
    similarity_df = pd.DataFrame()

if not hotels.empty:
    popular_hotels = hotels.groupby(['place', 'name']).size().reset_index(name='count')
    popular_hotels = popular_hotels.sort_values(by='count', ascending=False)
else:
    popular_hotels = pd.DataFrame()

# ==============================
# ROOT
# ==============================
@app.get("/")
def home():
    return {"message": "Voyage Analytics API Running"}

# ==============================
# PRICE PREDICTION
# ==============================
@app.post("/predict-price")
def predict_price(data: dict):
    if reg_model is None:
        return {"error": "Model not loaded"}

    try:
        df = pd.DataFrame([data])

        for col in reg_features:
            if col not in df.columns:
                df[col] = 0

        df = df[reg_features]

        prediction = reg_model.predict(df)[0]
        return {"predicted_price": float(prediction)}

    except Exception as e:
        return {"error": str(e)}

# ==============================
# GENDER PREDICTION
# ==============================
@app.post("/predict-gender")
def predict_gender(data: dict):
    if gender_model is None:
        return {"error": "Model not loaded"}

    try:
        df = pd.DataFrame([data])

        for col in gender_features:
            if col not in df.columns:
                df[col] = 0

        df = df[gender_features]

        prediction = gender_model.predict(df)[0]
        return {"predicted_gender": int(prediction)}

    except Exception as e:
        return {"error": str(e)}

# ==============================
# RECOMMEND TRIP
# ==============================
@app.get("/recommend-trip")
def recommend_trip(user_id: int):

    if popular_routes.empty:
        return {"error": "No data available"}

    if user_id not in similarity_df.index:
        destinations = popular_routes.head(3)['to'].tolist()
    else:
        similar_users = similarity_df[user_id].sort_values(ascending=False)[1:6]
        similar_ids = similar_users.index

        rec = user_route.loc[similar_ids].sum().sort_values(ascending=False)
        destinations = rec.head(3).index.tolist()

    result = {}

    for place in destinations:
        hotels_list = popular_hotels[popular_hotels['place'] == place].head(3)
        result[place] = hotels_list.to_dict(orient="records")

    return result
