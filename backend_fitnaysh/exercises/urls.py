from django.urls import path
from . import views

urlpatterns = [
    path('curl-video-stream', views.curl_video_feed , name='curl_video_stream'),
    path('bicep-curl/', views.curl_video, name='curl_video'),
    path('squat-video-stream', views.squat_video_feed , name='squat_video_stream'),
    path('squat/', views.squat_video, name='squat_video'),
    path('deadlift-video-stream', views.deadlift_video_feed , name='deadlift_video_stream'),
    path('deadlift/', views.deadlift_video, name='deadlift_video'),
]