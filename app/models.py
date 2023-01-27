from django.db import models
from django.contrib.auth.models import User

# Create your models here.

class Settings(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    max_auth = models.IntegerField()
    special_char = models.BooleanField()
    issue = models.BooleanField()
    journal_punct = models.BooleanField()
    duplicate_page = models.BooleanField()
