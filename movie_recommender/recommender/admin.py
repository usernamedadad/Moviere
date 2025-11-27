
# Register your models here.
from django.contrib import admin
from .models import Movie, Rating, UserRecommendation

@admin.register(Movie)
class MovieAdmin(admin.ModelAdmin):
    list_display = ['movie_id', 'title', 'genres']
    search_fields = ['title']

@admin.register(Rating)
class RatingAdmin(admin.ModelAdmin):
    list_display = ['user_id', 'movie', 'rating']
    list_filter = ['user_id', 'rating']

@admin.register(UserRecommendation)
class UserRecommendationAdmin(admin.ModelAdmin):
    list_display = ['user_id', 'created_at', 'updated_at']