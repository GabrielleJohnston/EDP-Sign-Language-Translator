#This file says look at whatever url the person requested and perform some
#functionality
from django.conf.urls import include, url
from django.contrib import admin

urlpatterns = [
    url(r'^admin/', admin.site.urls),
    url(r'^contact_us/', include('contact_us.urls')),
    url(r'^main_page/', include('main_page.urls')),
    url(r'^calibration/', include('calibration.urls')),
]
