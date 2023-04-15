from django.urls import path
from .views import bicep_curl_counter_view, video_feed

urlpatterns = [
    path('', bicep_curl_counter_view, name='counter'),
    path('video_feed/', video_feed, name='video_feed'),
]