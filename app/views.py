from django.shortcuts import render
from django.http import HttpResponse
from .forms import SearchForm


def index(request):
    return render(request, 'web/index.html')

def setting(request):
    # question = get_object_or_404(Question, pk=question_id)
    target = None
    results = None

    if request.method == 'POST':
        form = SearchForm(request.POST)
        if not torch.is_tensor(weighted_embeddings):
            error=True
            return render(request, 'web/search.html', {'form': form, 'target': target, 'results': results, 'error': error})
    else:
        form = SearchForm()

    return render(request, 'web/search.html', {'form': form, 'target': target, 'results': results})


def search(request):
    # question = get_object_or_404(Question, pk=question_id)
    target = None
    results = None
    error = False
    if request.method == 'POST':
        form = SearchForm(request.POST)
        if form.is_valid():
            print(form.cleaned_data['method'])

            return render(request, 'web/search.html', {'form': form, 'target': target, 'results': results, 'error': error})            

    else:
        form = SearchForm()

    return render(request, 'web/search.html', {'form': form, 'target': target, 'results': results})