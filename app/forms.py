from email.policy import default
from django import forms
from .models import Settings
from django.contrib.auth.models import User


MAX_AUTHOR = [(3,'3인'),(6, '6인'),(999,'제한없음')]
ADD_SPECIAL_CHAR = [(True, '특수문자 유지'),(False, '특수문자 삭제')]
ADD_ISSUE = [(True,'issue 유지'),(False, 'issue 삭제')]
ADD_JOURNAL_PUNCT = [(True,'학술지명 마침표 유지'),(False, '학술지명 마침표 삭제')]

class SearchForm(forms.Form):
    references = forms.CharField(widget=forms.Textarea(attrs={'style': 'width: 1500px; height: 800px'}))

# class SettingForm(forms.Form):
#     max_auth = forms.ChoiceField(widget=forms.Select, choices=MAX_AUTHOR)
#     special_char = forms.ChoiceField(widget=forms.Select, choices=ADD_SPECIAL_CHAR)
#     issue = forms.ChoiceField(widget=forms.Select, choices=ADD_ISSUE)
#     journal_punct = forms.ChoiceField(widget=forms.Select, choices=ADD_JOURNAL_PUNCT)

class UserForm(forms.ModelForm):
    class Meta:
        model = User
        fields = ['username', 'email', 'password']

class LoginForm(forms.ModelForm):
    class Meta:
        model = User
        fields = ['username', 'password'] # 로그인 시에는 유저이름과 비밀번호만 입력 받는다.

class SettingsForm(forms.ModelForm):
    # def __init__(self, *args, **kwargs):
    #      self.user = kwargs.pop('user',None)
    #      super(SettingsForm, self).__init__(*args, **kwargs)

    # max_auth = forms.ChoiceField(widget=forms.Select, choices=MAX_AUTHOR)
    # special_char = forms.CheckboxInput(attrs={'style':'width:20px;height:20px;'}),
    # issue = forms.ChoiceField(widget=forms.Select, choices=ADD_ISSUE),
    # journal_punct = forms.ChoiceField(widget=forms.Select, choices=ADD_JOURNAL_PUNCT),
    
    class Meta:
        model = Settings
        fields = ("max_auth", "special_char", "issue", "journal_punct")
        widgets = {
            'max_auth': forms.Select(attrs={'style': 'font-size: 20px'}, choices=MAX_AUTHOR),
            'special_char': forms.Select(attrs={'style': 'font-size: 20px'}, choices=ADD_SPECIAL_CHAR),
            'issue': forms.Select(attrs={'style': 'font-size: 20px'}, choices=ADD_ISSUE),
            'journal_punct': forms.Select(attrs={'style': 'font-size: 20px'}, choices=ADD_JOURNAL_PUNCT),
        }