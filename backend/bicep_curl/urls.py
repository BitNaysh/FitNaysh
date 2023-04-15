from django.urls import path
from .views import bicep_curl_view

urlpatterns = [
    path('bicep-curl-video/', bicep_curl_view, name='bicep_curl_video'),
]