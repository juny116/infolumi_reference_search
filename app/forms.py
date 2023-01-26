from email.policy import default
from django import forms

Q_DIGIT_CHOICES = (('1', '1'), ('2', '2'))

class SearchForm(forms.Form):
    method = forms.CharField(widget=forms.Textarea(attrs={'style': 'width: 1500px; height: 800px'}))

