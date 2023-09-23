"""A2SL URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from . import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('about/',views.about_view,name='about'),
    path('contact/',views.contact_view,name='contact'),
    path('animation/',views.animation_view,name='animation'),
    path('',views.home_view,name='home'),
    path('two_way/',views.two_way,name='two_way'),
    path('get_frames/',views.get_frames,name='get_frames'),
     path('live_feed/',views.live_feed,name='live_feed'),
]
