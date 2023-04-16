from django.urls import path
from .views import curl_video_feed, curl_video

urlpatterns = [
    path('curl-video-stream', curl_video_feed , name='curl_video_stream'),
    path('bicep-curl/', curl_video, name='curl_video'),
]