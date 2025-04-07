from rest_framework.decorators import api_view
from rest_framework.response import Response
from .utils import train_and_save_model, get_recommendations

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
