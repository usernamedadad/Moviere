from django.db import models

class Movie(models.Model):
    movie_id = models.IntegerField(unique=True, primary_key=True)
    title = models.CharField(max_length=255)
    genres = models.CharField(max_length=255, blank=True)
    
    def __str__(self):
        return self.title

class Rating(models.Model):
    user_id = models.IntegerField()  
    movie = models.ForeignKey(Movie, on_delete=models.CASCADE)
    rating = models.FloatField()
    timestamp = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        unique_together = ('user_id', 'movie')  

class UserRecommendation(models.Model):
    user_id = models.IntegerField(unique=True) 
    recommended_movies = models.JSONField()
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    def __str__(self):
        return f"Recommendations for user {self.user_id}"