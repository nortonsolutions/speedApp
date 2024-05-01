from django import forms
from .models import Video

# VideoForm with model
class VideoFormWithModel(forms.ModelForm):
    class Meta:
        model= Video
        fields= ["name", "videofile"]
        
# ref: Load existing:
# Create a form to edit an existing Article, but use
# POST data to populate the form.
# >>> a = Article.objects.get(pk=1)
# >>> f = ArticleForm(request.POST, instance=a)
# >>> f.save()

class VideoForm(forms.Form):
    name = forms.CharField(max_length=100)
    videofile = forms.FileField(label='Select a file')