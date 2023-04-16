from django.http import HttpResponse, StreamingHttpResponse
from django.shortcuts import render
import cv2
from .detect import fitNaysh
 
fn = fitNaysh()

# Create your views here.
def curl_video_feed(request):
    return StreamingHttpResponse(fn.bicep_curl_counter(),
                                 content_type='multipart/x-mixed-replace; boundary=frame')

def curl_video(request):
    context = {'counter': fn.counter, 'stage': fn.stage}
    return render(request, 'bicep_curl_video.html',context)
