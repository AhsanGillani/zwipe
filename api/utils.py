import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib
from firebase_admin import credentials, firestore, initialize_app
from google.cloud.firestore_v1.base_query import FieldFilter
from datetime import datetime
import os
from django.conf import settings
import numpy as np
import os
from django.conf import settings
    
# Initialize Firebase
cred = credentials.Certificate("api/credentials.json")
initialize_app(cred)
db = firestore.client()

def fetch_new_data(last_timestamp):

    print(last_timestamp)
    docs = db.collection('motors')\
        .where(filter=FieldFilter("createdDate", ">", last_timestamp))\
        .stream()

    data = []
    for doc in docs:
        item = doc.to_dict()
        item['doc_id'] = doc.id
        item['createdDate'] = item['createdDate']
        data.append(item)
  
    return pd.DataFrame(data)


def combine_features(row):
    return f"{row['title']} {row['fuelType']} {row['carMake']} {row['carModel']} {row['bodyType']} {row['transmissionType']} {row['engineSize']}L {row['yearOfManufacture']} {row['horsePower']}HP {row['steeringSide']} {row['regionalSpec']} {row['category']}"


def train_and_save_model():


    try:
        model_dir = os.path.join(settings.BASE_DIR, 'api', 'recommender_model')
        old_df = joblib.load(os.path.join(model_dir, 'vehicle_data.pkl'))
        print("Length of old dataset", len(old_df))

        # Ensure createdDate is in datetime format
        old_df['createdDate'] = pd.to_datetime(old_df['createdDate'], errors='coerce')
        last_timestamp = old_df['createdDate'].max()

        # If last_timestamp is NaT, use default datetime
        if pd.isna(last_timestamp):
            print("last_timestamp is NaT. Using default datetime.")
            last_timestamp = pd.Timestamp("2000-01-01 00:00:00")

    except Exception as e:
        print("Error loading existing model:", e)
        old_df = pd.DataFrame()
        last_timestamp = pd.Timestamp("2000-01-01 00:00:00")  # Default starting point

    # Fetch new data from Firestore
    new_data = fetch_new_data(last_timestamp)

    if new_data.empty:
        return "No new data to train."

    # Merge old and new data
    df = pd.concat([old_df, new_data], ignore_index=True)
    df.fillna('', inplace=True)
    df['text'] = df.apply(combine_features, axis=1)

    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['text'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    df = df.drop(columns=['relatedUser'], errors='ignore')

    # Save everything
    joblib.dump(df, os.path.join(model_dir, 'vehicle_data.pkl'))
    joblib.dump(tfidf, os.path.join(model_dir, 'vectorizer.pkl'))
    joblib.dump(cosine_sim, os.path.join(model_dir, 'cosine_sim.pkl'))

    return f"Model trained on {len(df)} total items."



def get_recommendations(title, top_n=4):


    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(base_dir, 'recommender_model')

    try:
        df = joblib.load(os.path.join(model_dir, 'vehicle_data.pkl'))
        cosine_sim = joblib.load(os.path.join(model_dir, 'cosine_sim.pkl'))
    except Exception as e:
        print("Model loading failed:", str(e))
        return {}

    df['title'] = df['title'].astype(str)
    indices = pd.Series(df.index, index=df['title'])

    if title not in indices:
        print(f"Title '{title}' not found in dataset.")
        return {}

    try:
        if isinstance(indices[title], pd.Series):
            idx = indices[title].iloc[0]
        else:
            idx = indices[title]

        # Get the query product info
        query_product_info = df.iloc[idx][['doc_id', 'title']].to_dict()

        # Get similarity scores
        sim_scores = list(enumerate(cosine_sim[int(idx)]))
        sim_scores = sorted(sim_scores, key=lambda x: float(x[1]), reverse=True)[1:top_n + 1]

        vehicle_indices = [i[0] for i in sim_scores if isinstance(i[0], (int, np.integer)) and i[0] < len(df)]

        recommendations = df.iloc[vehicle_indices][['doc_id', 'title', 'carMake', 'carModel']]
        recommendations = recommendations.drop_duplicates(subset='title')

        return {
            "queryProduct": query_product_info,
            "recommendations": recommendations.to_dict(orient='records')
        }

    except Exception as e:
        print("Error during recommendation processing:", str(e))
        return {}
