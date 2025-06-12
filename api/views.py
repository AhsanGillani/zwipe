from rest_framework.decorators import api_view
from rest_framework.response import Response
from .utils import train_and_save_model, get_recommendations
import joblib
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from firebase_admin import firestore
from datetime import datetime

# Initialize Firebase
db = firestore.client()

def clean_path(path):
    return path[1:] if path.startswith('/') else path

@api_view(['POST'])
def train_model(request):
    msg = train_and_save_model()
    return Response({'message': msg})

@api_view(['GET'])
def recommend_vehicle(request):
    title = request.GET.get('title', None)
    related_user = request.GET.get('relatedUser', None)
    
    if not title:
        return Response({"error": "Missing vehicle title"}, status=400)
    if not related_user:
        return Response({"error": "Missing relatedUser"}, status=400)

    recommendations = get_recommendations(title)
    if not recommendations:
        return Response({"message": "No recommendations found."})
    
    # Get the query product info
    query_product = recommendations.get("queryProduct", {})
    query_doc_id = query_product.get("doc_id")
    
    # Store recommendations in Firebase
    recommendations_collection = db.collection('recommendations')
    
    for rec in recommendations.get("recommendations", []):
        # Build Firestore reference paths
        related_user_ref = db.document(clean_path(related_user))
        related_listing_ref = db.document(f"motors/{query_doc_id.lstrip('/')}")
        recommended_listing_ref = db.document(f"motors/{rec['doc_id'].lstrip('/')}")

        # Check if recommendation already exists
        existing_docs = recommendations_collection \
            .where('relatedUser', '==', related_user_ref) \
            .where('relatedListing', '==', related_listing_ref) \
            .where('recommendedListing', '==', recommended_listing_ref) \
            .limit(1) \
            .stream()

        if not list(existing_docs):
            recommendations_collection.add({
                'relatedUser': related_user_ref,
                'relatedListing': related_listing_ref,
                'recommendedListing': recommended_listing_ref,
                'createdDate': datetime.now()
            })
    
    return Response({"recommendations": recommendations})

@api_view(['POST'])
def reset_model(request):
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_dir = os.path.join(base_dir, 'recommender_model')

        # Load existing vehicle_data.pkl to get column structure
        df_path = os.path.join(model_dir, 'vehicle_data.pkl')
        if os.path.exists(df_path):
            df = joblib.load(df_path)
            df = df.iloc[0:0]  # Remove all rows but keep column structure
            joblib.dump(df, df_path)
        else:
            return Response({"error": "vehicle_data.pkl file not found."}, status=404)

        # Replace cosine_sim.pkl with an empty 0x0 matrix
        cosine_path = os.path.join(model_dir, 'cosine_sim.pkl')
        if os.path.exists(cosine_path):
            empty_cosine = np.zeros((0, 0))
            joblib.dump(empty_cosine, cosine_path)
        else:
            return Response({"error": "cosine_sim.pkl file not found."}, status=404)

        return Response({"message": "All data rows cleared, headers preserved."}, status=200)

    except Exception as e:
        return Response({"error": str(e)}, status=500)
    



@api_view(['DELETE'])
def delete_vehicle(request):
    doc_id = request.query_params.get('doc_id')
    if not doc_id:
        return Response({"error": "doc_id is required."}, status=400)
    
    try:
        # Load existing data
        df = joblib.load("api/recommender_model/vehicle_data.pkl")
        
        # Check if doc_id exists
        if doc_id not in df['doc_id'].values:
            return Response({"error": "Document ID not found."}, status=404)
        
        # Remove the record
        df = df[df['doc_id'] != doc_id].reset_index(drop=True)
        
        # Recreate text feature
        df['text'] = df.apply(lambda row: f"{row['title']} {row['fuelType']} {row['carMake']} {row['carModel']} {row['bodyType']} {row['transmissionType']} {row['engineSize']}L {row['yearOfManufacture']} {row['horsePower']}HP {row['steeringSide']} {row['regionalSpec']} {row['category']}", axis=1)

        # Re-train TF-IDF and Cosine Similarity
        tfidf = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf.fit_transform(df['text'])
        cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

        # Save updated files
        joblib.dump(df, "api/recommender_model/vehicle_data.pkl")
        joblib.dump(tfidf, "api/recommender_model/vectorizer.pkl")
        joblib.dump(cosine_sim, "api/recommender_model/cosine_sim.pkl")
        
        return Response({"message": "Vehicle deleted and model updated."})
    
    except Exception as e:
        return Response({"error": str(e)}, status=500)
