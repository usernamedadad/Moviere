from django.urls import path, include
from rest_framework.routers import DefaultRouter
from . import views

router = DefaultRouter()
router.register(r'movies', views.MovieViewSet)
router.register(r'ratings', views.RatingViewSet) 
router.register(r'recommendations', views.RecommendationViewSet, basename='recommendations')

urlpatterns = [
    # API路由
    path('api/', include(router.urls)),
    
    # 前端页面路由
    path('', views.index, name='index'),
    path('movies/', views.movie_list, name='movie_list'),
    path('recommendations/', views.user_recommendations, name='user_recommendations'),
    path('search/', views.search_movies, name='search_movies'),
]