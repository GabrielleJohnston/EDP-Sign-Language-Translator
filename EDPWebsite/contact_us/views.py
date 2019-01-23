from django.shortcuts import render
#functions that take a user request and give them back something
# Create your views here.
from django.http import HttpResponse

def index(request):
    return HttpResponse('<h1>This is the contact us page</h1>')
