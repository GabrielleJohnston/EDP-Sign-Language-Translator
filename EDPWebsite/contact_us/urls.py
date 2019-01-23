from django.conf.urls import url
from . import views #the dot means look at the current directory i am in

urlpatterns = [
    url(r'^$', views.index, name='index'),
]
