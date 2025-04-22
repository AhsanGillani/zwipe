from rest_framework.decorators import api_view
from rest_framework.response import Response
from .utils import train_and_save_model, get_recommendations
import joblib
import os

import numpy as np
# Initialize Firebase

@api_view(['POST'])
def train_model(request):
    msg = train_and_save_model()
    return Response({'message': msg})

@api_view(['GET'])
def recommend_vehicle(request):
    title = request.GET.get('title', None)
    if not title:
        return Response({"error": "Missing vehicle title"}, status=400)

    recommendations = get_recommendations(title)
    if not recommendations:
        return Response({"message": "No recommendations found."})
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
