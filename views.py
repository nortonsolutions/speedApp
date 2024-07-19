from django.http import HttpResponse,HttpResponseRedirect
from django.shortcuts import render
import asyncio

from .forms import VideoForm, VideoFormWithModel
from .speedApp import process_new_video

# from asgiref.sync import iscoroutinefunction, markcoroutinefunction

# from myapp.models import Video

from django.db.models.signals import post_save
from django.dispatch import receiver
from .models import Video

@receiver(post_save, sender=Video)
def notify_user(sender, instance, created, **kwargs):
    if created:
        # Notification logic here
        # ...
        print("Video created")

# def showvideo(request):
#     lastvideo= Video.objects.last()
#     videofile= lastvideo.videofile
#     form= VideoForm(request.POST or None, request.FILES or None)
#     if form.is_valid():
#         form.save()
#     context= {
#         'videofile': videofile,
#         'form': form
#     }
#     return render(request, 'videos.html', context)

videofile = "media/temp.mp4"

# def upload_file(request):
#     if request.method == "POST":
#         form = VideoForm(request.POST, request.FILES)
#         if form.is_valid():
#             handle_uploaded_file(request.FILES["file"])
#             return HttpResponseRedirect("/success/url/")
#     else:
#         form = VideoForm()
#     return render(request, "upload.html", {"form": form})

# Handle file upload manually, without using a model
def handle_uploaded_file(f) -> bool:
    # if the open operation fails, return false, otherwise true
    with open(videofile, "wb+") as destination:
        for chunk in f.chunks():
            destination.write(chunk)
        
    return not destination.closed

def index(request):
    # print("index")
    # if this is a POST request we need to process the form data
    if request.method == "POST":
        form = VideoFormWithModel(request.POST, request.FILES)
        if form.is_valid():
            print("form is valid")
            # With model:
            if not form.save():
            # old method: handle the file upload manually
            # if err := handle_uploaded_file(request.FILES["videofile"]):
                return HttpResponse(f"file upload failed: {str(form.errors)}")

            print("file upload successful")
            
            # Process the video with the model
            video_filename = asyncio.run(process_new_video(filename="output.webm"))
            print(f"video_filename: {video_filename}")
            
            # force reload on this rendering:
            context = {"form": form, "app_name": "speedApp", "video_filename": video_filename}
            return render(request, "simple_upload.html", context)

        else:
            print("form not valid")
    else:
        # old method: create a context to represent the form
        # context = {
        #     "videofile": "nortonTrafficDetection.mp4",
        # }
        form = VideoFormWithModel()
        return render(request, "simple_upload.html", {"form": form, "app_name": "speedApp"})

# markcoroutinefunction(index)
# def static(request):
#     return render(request, request.path[1:].html, {})

# from django.views.generic.edit import FormView
# from .forms import FileFieldForm

# class FileFieldFormView(FormView):
#     form_class = FileFieldForm
#     template_name = "upload.html"  # Replace with your template.
#     success_url = "..."  # Replace with your URL or reverse().

#     def post(self, request, *args, **kwargs):
#         form_class = self.get_form_class()
#         form = self.get_form(form_class)
#         if form.is_valid():
#             return self.form_valid(form)
#         else:
#             return self.form_invalid(form)

#     def form_valid(self, form):
#         files = form.cleaned_data["file_field"]
#         for f in files:
#             ...  # Do something with each file.
#         return super().form_valid()
