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

def squat_video_feed(request):
    return StreamingHttpResponse(fn.squat_counter(),
                                 content_type='multipart/x-mixed-replace; boundary=frame')

def squat_video(request):
    context = {'counter': fn.counter, 
               'stage': fn.stage,
               'back_form': fn.back_form,
               'heelisGrounded': fn.heelIsGrounded,
               'notlowSquat': fn.not_low_squat,}
    return render(request, 'squat_video.html',context)

def deadlift_video_feed(request):
    return StreamingHttpResponse(fn.deadlift_counter(),
                                 content_type='multipart/x-mixed-replace; boundary=frame')

def deadlift_video(request):
    context = {'counter': fn.counter, 'stage': fn.stage, 'back_form': fn.back_form}
    return render(request, 'deadlift_video.html',context)
