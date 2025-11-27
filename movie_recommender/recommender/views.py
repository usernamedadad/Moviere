from rest_framework import viewsets, status
from rest_framework.decorators import action
from rest_framework.response import Response
from django.shortcuts import get_object_or_404
from .models import Movie, Rating, UserRecommendation
from .serializers import MovieSerializer, RatingSerializer, UserRecommendationSerializer
from .utils import recommendation_engine

class MovieViewSet(viewsets.ModelViewSet):
    queryset = Movie.objects.all()
    serializer_class = MovieSerializer
    
    @action(detail=False, methods=['get'])
    def search(self, request):
        """搜索电影"""
        query = request.query_params.get('q', '')
        if query:
            movies = Movie.objects.filter(title__icontains=query)
        else:
            movies = Movie.objects.all()[:50]  # 默认返回前50部电影
        
        serializer = self.get_serializer(movies, many=True)
        return Response(serializer.data)

class RatingViewSet(viewsets.ModelViewSet):
    queryset = Rating.objects.all()
    serializer_class = RatingSerializer
    
    def get_queryset(self):
        """根据用户ID过滤评分"""
        queryset = Rating.objects.all()
        user_id = self.request.query_params.get('user_id')
        if user_id is not None:
            queryset = queryset.filter(user_id=user_id)
        return queryset
    
    @action(detail=False, methods=['post'])
    def rate_movie(self, request):
        """用户给电影评分"""
        user_id = request.data.get('user_id')
        movie_id = request.data.get('movie_id')
        rating = request.data.get('rating')
        
        if not all([user_id, movie_id, rating]):
            return Response(
                {'error': '缺少必要参数: user_id, movie_id, rating'}, 
                status=status.HTTP_400_BAD_REQUEST
            )
        
        movie = get_object_or_404(Movie, movie_id=movie_id)
        rating_obj, created = Rating.objects.update_or_create(
            user_id=user_id,
            movie=movie,
            defaults={'rating': rating}
        )
        
        serializer = RatingSerializer(rating_obj)
        return Response(serializer.data)

class RecommendationViewSet(viewsets.ViewSet):
    
    @action(detail=False, methods=['get'])
    def get_recommendations(self, request):
        """获取用户的电影推荐"""
        user_id = request.query_params.get('user_id')
        if not user_id:
            return Response(
                {'error': '需要提供user_id参数'}, 
                status=status.HTTP_400_BAD_REQUEST
            )
        
        try:
            user_id = int(user_id)
        except ValueError:
            return Response(
                {'error': 'user_id必须是整数'}, 
                status=status.HTTP_400_BAD_REQUEST
            )
        
        # 检查是否已有推荐
        try:
            user_rec = UserRecommendation.objects.get(user_id=user_id)
            serializer = UserRecommendationSerializer(user_rec)
            return Response(serializer.data)
        except UserRecommendation.DoesNotExist:
            pass
        
        # 生成新推荐
        recommended_movie_ids = recommendation_engine.get_recommendations(user_id)
        
        # 保存推荐结果
        user_rec = UserRecommendation.objects.create(
            user_id=user_id,
            recommended_movies=recommended_movie_ids
        )
        
        serializer = UserRecommendationSerializer(user_rec)
        return Response(serializer.data)
    
    @action(detail=False, methods=['post'])
    def refresh_recommendations(self, request):
        """刷新用户的电影推荐"""
        user_id = request.data.get('user_id')
        if not user_id:
            return Response(
                {'error': '需要提供user_id参数'}, 
                status=status.HTTP_400_BAD_REQUEST
            )
        
        # 生成新推荐
        recommended_movie_ids = recommendation_engine.get_recommendations(user_id)
        
        # 更新或创建推荐记录
        user_rec, created = UserRecommendation.objects.update_or_create(
            user_id=user_id,
            defaults={'recommended_movies': recommended_movie_ids}
        )
        
        serializer = UserRecommendationSerializer(user_rec)
        return Response(serializer.data)
    
    
from django.shortcuts import render
from django.core.paginator import Paginator

def index(request):
    """首页"""
    return render(request, 'index.html')

def movie_list(request):
    """电影列表页"""
    return render(request, 'movie_list.html')

def user_recommendations(request):
    """用户推荐页"""
    return render(request, 'user_recommendations.html')

def search_movies(request):
    """搜索电影"""
    query = request.GET.get('q', '')
    return render(request, 'search_results.html', {'query': query})
def search_movies(request):
    """搜索电影页面"""
    query = request.GET.get('q', '')
    return render(request, 'search_results.html', {'query': query})