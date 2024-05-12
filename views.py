from django.http import HttpResponse,HttpResponseRedirect
from django.shortcuts import render

from .forms import VideoForm, VideoFormWithModel
from .speed_predict import process_new_video

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
        # create a form instance and populate it with data from the request:
        form = VideoFormWithModel(request.POST, request.FILES)
        if form.is_valid():
            # With model:
            if not form.save():
            # old method: handle the file upload manually
            # if err := handle_uploaded_file(request.FILES["videofile"]):
                return HttpResponse(f"file upload failed: {str(form.errors)}")

            print("file upload successful")
            
            # Process the video with the model
            err = process_new_video(form.instance.videofile.path)
            print(err)
            
            return render(request, "simple_upload.html", {"form": form})
        else:
            print("form not valid")
    else:
        # old method: create a context to represent the form
        # context = {
        #     "videofile": "nortonTrafficDetection.mp4",
        # }
        form = VideoFormWithModel()
        return render(request, "simple_upload.html", {"form": form})

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