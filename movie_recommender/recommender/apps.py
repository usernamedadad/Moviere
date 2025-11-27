from django.apps import AppConfig

class RecommenderConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'recommender'
    
    def ready(self):
        # 导入信号处理器等
        pass