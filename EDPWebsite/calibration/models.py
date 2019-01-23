from django.db import models
#This is a model for both my database and my python code
#make a class - each variable in class is converted to a column in database
# Create your models here.
class Picture(models.Model):
    coordNum = models.IntegerField()
    x = models.IntegerField()
    y = models.IntegerField()
    #image = models.CharField(max_length=1000)
