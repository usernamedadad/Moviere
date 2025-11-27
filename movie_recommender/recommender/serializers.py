from rest_framework import serializers
from .models import Movie, Rating, UserRecommendation

class MovieSerializer(serializers.ModelSerializer):
    class Meta:
        model = Movie
        fields = ['movie_id', 'title', 'genres']

class RatingSerializer(serializers.ModelSerializer):
    movie_title = serializers.CharField(source='movie.title', read_only=True)
    
    class Meta:
        model = Rating
        fields = ['id', 'user_id', 'movie', 'movie_title', 'rating', 'timestamp']

class UserRecommendationSerializer(serializers.ModelSerializer):
    recommended_movies_details = serializers.SerializerMethodField()
    
    class Meta:
        model = UserRecommendation
        fields = ['user_id', 'recommended_movies', 'recommended_movies_details', 'created_at', 'updated_at']
    
    def get_recommended_movies_details(self, obj):
        # 获取推荐电影的详细信息
        movie_ids = obj.recommended_movies
        movies = Movie.objects.filter(movie_id__in=movie_ids)
        return MovieSerializer(movies, many=True).data