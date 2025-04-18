import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib
from firebase_admin import credentials, firestore, initialize_app
from google.cloud.firestore_v1.base_query import FieldFilter
from datetime import datetime
import os
# Initialize Firebase
cred = credentials.Certificate("credentials.json")
initialize_app(cred)
db = firestore.client()

def fetch_new_data(last_timestamp):

    print(last_timestamp)
    docs = db.collection('motors')\
        .where(filter=FieldFilter("createdDate", ">=", last_timestamp))\
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
        old_df = joblib.load("api/recommender_model/vehicle_data.pkl")
        print(len(old_df))
        last_timestamp = old_df['createdDate'].max()
        
    except:
        old_df = pd.DataFrame()
        last_timestamp = "2025-04-06 11:41:06.329398+00:00"



    new_data = fetch_new_data(last_timestamp)
    print(new_data)
    print(len(new_data))
    if new_data.empty:
        return "No new data to train."

    df = pd.concat([old_df, new_data], ignore_index=True)
    df.fillna('', inplace=True)
    df['text'] = df.apply(combine_features, axis=1)

    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['text'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    joblib.dump(df, "api/recommender_model/vehicle_data.pkl")
    joblib.dump(tfidf, "api/recommender_model/vectorizer.pkl")
    joblib.dump(cosine_sim, "api/recommender_model/cosine_sim.pkl")
    return f"Model trained on {len(df)} total items."





def get_recommendations(title, top_n=5):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(base_dir, 'recommender_model')
    try:
        df = joblib.load(os.path.join(model_dir, 'vehicle_data.pkl'))
        cosine_sim = joblib.load(os.path.join(model_dir, 'cosine_sim.pkl'))
    except:
        return []

    indices = pd.Series(df.index, index=df['title']).drop_duplicates()
    if title not in indices:
        return []

    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
    vehicle_indices = [i[0] for i in sim_scores]
    return df.iloc[vehicle_indices][['doc_id', 'title', 'carMake', 'carModel']].to_dict(orient='records')
