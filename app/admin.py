from django.contrib import admin

# Register your models here.
from .models import Settings, User

class SettingsAdmin(admin.ModelAdmin):
    fields = ['user', 'max_auth', 'special_char', 'issue', 'journal_punct']

admin.site.register(Settings, SettingsAdmin)

