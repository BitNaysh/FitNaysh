from django.urls import path
from .views import bicep_curl_view, bicep_curl_video

urlpatterns = [
    path('bicep-curl-video/', bicep_curl_video, name='bicep_curl_video'),
]