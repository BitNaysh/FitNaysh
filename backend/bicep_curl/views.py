from django.shortcuts import render
from django.http import JsonResponse
from .detect import fitNaysh
import base64
import cv2
import json

# Create your views here.
def bicep_curl_view(request):
    bicep_curl = fitNaysh()
    bicep_curl.bicep_curl_counter()
    counter = bicep_curl.counter

    retval, buffer = cv2.imencode('.jpg', bicep_curl.image)
    jpg_as_text = base64.b64encode(buffer).decode()

    return render(request, 'bicep_curl_video.html', {'counter': counter, 'image': jpg_as_text})

def bicep_curl_video(request):
    bicep_curl = fitNaysh()
    bicep_curl.bicep_curl_counter()
    
    # encode the image as base64 string
    retval, buffer = cv2.imencode('.jpg', bicep_curl.image)
    img_str = base64.b64encode(buffer).decode()

    # return the image as a JSON response
    return JsonResponse({'img_str': img_str})