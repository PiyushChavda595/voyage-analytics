from fastapi import FastAPI
import pandas as pd
import joblib

app = FastAPI(title="Voyage Analytics API")

# ==============================
# LOAD MODELS
# ==============================
reg_model = joblib.load("models/final_clean_model.pkl")
reg_features = joblib.load("models/features.pkl")

gender_model = joblib.load("models/gender_model_final.pkl")
gender_features = joblib.load("models/gender_features.pkl")

# ==============================
# LOAD DATA (for recommender)
# ==============================
flights = pd.read_csv("data/flights.csv")
hotels = pd.read_csv("data/hotels.csv")

# Precompute
popular_routes = flights.groupby(['from', 'to']).size().reset_index(name='count')
popular_routes = popular_routes.sort_values(by='count', ascending=False)

popular_hotels = hotels.groupby(['place', 'name']).size().reset_index(name='count')
popular_hotels = popular_hotels.sort_values(by='count', ascending=False)

user_route = flights.groupby(['userCode', 'to']).size().unstack(fill_value=0)

from sklearn.metrics.pairwise import cosine_similarity
similarity = cosine_similarity(user_route)

similarity_df = pd.DataFrame(
    similarity,
    index=user_route.index,
    columns=user_route.index
)

# ==============================
# ROOT
# ==============================
@app.get("/")
def home():
    return {"message": "Voyage Analytics API Running"}

# ==============================
# REGRESSION API
# ==============================
@app.post("/predict-price")
def predict_price(data: dict):
    try:
        # Convert to DataFrame
        df = pd.DataFrame([data])

        # Add missing columns
        for col in reg_features:
            if col not in df.columns:
                df[col] = 0

        # Keep only required columns in correct order
        df = df[reg_features]

        # Predict
        prediction = reg_model.predict(df)[0]

        return {"predicted_price": float(prediction)}

    except Exception as e:
        return {"error": str(e)}

# ==============================
# GENDER API
# ==============================
@app.post("/predict-gender")
def predict_gender(data: dict):
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
# RECOMMEND DESTINATIONS
# ==============================
@app.get("/recommend-destinations")
def recommend_destinations(user_id: int):
    
    if user_id not in similarity_df.index:
        return popular_routes.head(5).to_dict(orient="records")
    
    similar_users = similarity_df[user_id].sort_values(ascending=False)[1:6]
    similar_ids = similar_users.index
    
    rec = user_route.loc[similar_ids].sum().sort_values(ascending=False).head(5)
    
    return rec.to_dict()

# ==============================
# RECOMMEND HOTELS
# ==============================
@app.get("/recommend-hotels")
def recommend_hotels(place: str):
    
    result = popular_hotels[popular_hotels['place'] == place]
    
    if result.empty:
        result = popular_hotels.head(5)
    
    return result.to_dict(orient="records")

# ==============================
# FULL TRIP RECOMMENDER
# ==============================
@app.get("/recommend-trip")
def recommend_trip(user_id: int):
    
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