from django.shortcuts import render, redirect
from django.http import HttpResponse
from .forms import SearchForm, LoginForm, SettingsForm
from .models import Settings
from .utils import *

def index(request):
    return render(request, 'web/index.html')


def settings(request):
    if not request.user.is_authenticated:
        return redirect('index')
    # try:
    item, created = Settings.objects.get_or_create(user=request.user, 
        defaults={'max_auth': 3, 'special_char': True, 'issue': True, 'journal_punct': True})

    success = False
    if request.method == 'POST':
        form = SettingsForm(request.POST, instance=item)
        if form.is_valid():
            form.save()
        else:
            print(form.errors.as_data()) 
    else:
        form = SettingsForm(instance=item)

    return render(request, 'web/settings.html', {'form': form, 'success': success})


def search(request):
    if not request.user.is_authenticated:
        return redirect('index')
    # question = get_object_or_404(Question, pk=question_id)
    target = None
    results = None
    error = False

    if request.method == 'POST':
        item, created = Settings.objects.get_or_create(user=request.user, 
            defaults={'max_auth': 3, 'special_char': True, 'issue': True, 'journal_punct': True, 'duplicate_page': True})
        form = SearchForm(request.POST)
        if form.is_valid():
            uid_dict, original_list = SearchPubmedWeb(form.cleaned_data['references'].splitlines(), item)
            revised_text = FetchPubmedAPI(uid_dict, original_list, item)
            form = SearchForm(initial={'references': revised_text})
            return render(request, 'web/search.html', {'form': form, 'target': target, 'results': revised_text, 'error': error})            
    else:
        form = SearchForm()

    return render(request, 'web/search.html', {'form': form, 'target': target, 'results': results})