from django.urls import path

from . import views
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path("", views.index, name="index"),
    # path("v1/", views.index, name="index"),
    # path("static/", views.static, name="static"),
    # path("upload/", views.upload, name="upload"),
    
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
