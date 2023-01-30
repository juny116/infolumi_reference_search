import xmltodict
import requests
from bs4 import BeautifulSoup
import urllib.parse
import time
import truecase
from difflib import SequenceMatcher
import re


def LowerBesidesProper(text):
    truecase_text = truecase.get_true_case(text)
    if truecase_text == text:
        changed = False
    else:
        changed = True

    return text, changed

def GetMedlinePage(start, end):
    if start == end:
        return start

    start_with_zero = (len(end) - len(start))*'0' + start
    same_cnt = 0
    for e, s in zip(end, start_with_zero):
        if e == s:
            same_cnt += 1
        else:
            break
    medline_page = f":{start}-{end[same_cnt:]}"
    return medline_page


def SearchPubmedWeb(data, setting):
    original_list = []
    uid_dict = {}
    for i, line in enumerate(data):
        line = line.split('.', 1)[1].strip()
        original_list.append(line)
        authors, line = line.split('.', 1)
        # authors = authors.split(',')
        try:
            title, potential_title, line = re.split(r'[.|?|:]', line, maxsplit=2)
        except:
            title, potential_title = line.split('.', 1)
            print(i, title)
        title = title.strip().lower()
        potential_title = potential_title.strip().lower()

        found = False
        # search with title
        query = urllib.parse.quote_plus(authors + '. ' + title)
        url = f"https://pubmed.ncbi.nlm.nih.gov/?term={query}&size=100"
        r = requests.get(url)
        if r.status_code == 200:
            html = r.text
            soup = BeautifulSoup(html, 'html.parser')
            if soup.select_one('.current-id'):
                current_title = soup.select_one("#full-view-heading > h1").text.strip().lower()
                if len(re.split(r'[.|?|:]', current_title)) > 1:
                    title = f"{title}. {potential_title}"
                if SequenceMatcher(None, current_title, title).ratio() > 0.9:
                    uid_dict[i] = soup.select_one('.current-id').text
                    found = True
                else:
                    print("not exact match")
                    print(current_title)
                    print(title)
                    uid_dict[i] = '-1'
            elif soup.select_one('#search-results > section > div.search-results-chunks > div > article:nth-child(2) > div.docsum-wrap > div.docsum-content > div.docsum-citation.full-citation > span:nth-child(5) > span'):
                answers = soup.select("#search-results > section > div.search-results-chunks > div > article")
                for ans in answers:
                    current_title = ans.select_one("div.docsum-wrap > div.docsum-content > a").text.strip()[:-1].lower()
                    if len(re.split(r'[.|?|:]', current_title)) > 1:
                        temp_title = f"{title}. {potential_title}"
                    else:
                        temp_title = title
                    if SequenceMatcher(None, current_title, temp_title).ratio() > 0.9:
                        found = True
                        uid_dict[i] = ans.select_one("div.docsum-wrap > div.docsum-content > div.docsum-citation.full-citation > span:nth-child(5) > span").text
                        break
                if not found:
                    uid_dict[i] = '-1'
            else:
                uid_dict[i] = '-1'
        else : 
            print(r.status_code)
        time.sleep(0.5)
        # search by title + potential title
        if not found:
            query = urllib.parse.quote_plus(authors + '. ' + title + '. ' + potential_title)
            url = f"https://pubmed.ncbi.nlm.nih.gov/?term={query}&size=100"
            r = requests.get(url)
            if r.status_code == 200:
                html = r.text
                soup = BeautifulSoup(html, 'html.parser')
                if soup.select_one('.current-id'):
                    current_title = soup.select_one("#full-view-heading > h1").text.strip().lower()
                    if len(re.split(r'[.|?|:]', current_title)) > 1:
                        title = f"{title}. {potential_title}"
                    if SequenceMatcher(None, current_title, title).ratio() > 0.9:
                        uid_dict[i] = soup.select_one('.current-id').text
                        found = True
                    else:
                        print("not exact match")
                        print(current_title)
                        print(title)
                        uid_dict[i] = '-1'
                elif soup.select_one('#search-results > section > div.search-results-chunks > div > article:nth-child(2) > div.docsum-wrap > div.docsum-content > div.docsum-citation.full-citation > span:nth-child(5) > span'):
                    answers = soup.select("#search-results > section > div.search-results-chunks > div > article")
                    for ans in answers:
                        current_title = ans.select_one("div.docsum-wrap > div.docsum-content > a").text.strip()[:-1].lower()
                        if len(re.split(r'[.|?|:]', current_title)) > 1:
                            temp_title = f"{title}. {potential_title}"
                        else:
                            temp_title = title
                        if SequenceMatcher(None, current_title, temp_title).ratio() > 0.9:
                            found = True
                            uid_dict[i] = ans.select_one("div.docsum-wrap > div.docsum-content > div.docsum-citation.full-citation > span:nth-child(5) > span").text
                            break
                    if not found:
                        uid_dict[i] = '-1'
                else:
                    uid_dict[i] = '-1'
            else : 
                print(r.status_code)

    return uid_dict, original_list


def FetchPubmedAPI(uid_dict, original_list, setting):
    uid_list = [uid for uid in uid_dict.values() if uid != '-1']
    uid_string = ','.join(uid_list)
    # url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed&api_key={api_key}&id={uid_string}"
    url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed&id={uid_string}"
    r = requests.get(url)
    results = xmltodict.parse(r.text)

    revised_list = []
    for k, v in uid_dict.items():
        max_authors = False
        if v == '-1':
            revised_list.append(f'{k+1}. {original_list[k]} NOT FOUND')
            # print(f'{k+1}. {original_list[k]} NOT FOUND')
        else:
            uid_index = uid_list.index(v)
            revised = f'{k+1}. '
            authors = results['PubmedArticleSet']['PubmedArticle'][uid_index]['MedlineCitation']['Article']['AuthorList']['Author']
            title = results['PubmedArticleSet']['PubmedArticle'][uid_index]['MedlineCitation']['Article']['ArticleTitle']
            journal = results['PubmedArticleSet']['PubmedArticle'][uid_index]['MedlineCitation']['Article']['Journal']
            page = results['PubmedArticleSet']['PubmedArticle'][uid_index]['MedlineCitation']['Article']['Pagination']

            if len(authors) > max(6, setting.max_auth):
                authors = authors[:setting.max_auth]
                max_authors = True

            if type(authors) == list:
                authors = [f"{author['LastName']} {author['Initials']}" for author in authors]
            elif type(authors) == dict:
                authors = [f"{authors['LastName']} {authors['Initials']}"]
            revised += ', '.join(authors)
            if max_authors:
                revised += ', et al'

            proper_nouns, changed = LowerBesidesProper(title)

            revised += f'. {title} '
            if setting.journal_punct:
                revised += f"{journal['ISOAbbreviation']}. "
            else:
                revised += f"{journal['ISOAbbreviation']} "
            revised += f"{journal['JournalIssue']['PubDate']['Year']}"
            if journal['JournalIssue'].get('Volume'):
                revised += f";{journal['JournalIssue']['Volume']}"
            if journal['JournalIssue'].get('Issue') and setting.issue:
                if journal['JournalIssue'].get('Volume') is None:
                    revised += f";({journal['JournalIssue']['Issue']})"
                else:
                    revised += f"({journal['JournalIssue']['Issue']})"
            if setting.duplicate_page:
                if page.get('StartPage') != page.get('EndPage'):
                    revised += f":{page.get('StartPage')}-{page.get('EndPage')}"
                else:
                    revised += f":{page.get('StartPage')}"
            else:
                if page.get('MedlinePgn'):
                    revised += f":{page.get('MedlinePgn')}"
                elif page.get('EndPage'):
                    revised += GetMedlinePage(page.get('StartPage'), page.get('EndPage'))
                else:
                    revised += f":{page.get('StartPage')}"
            revised += '.'

            revised_list.append(revised)
    
    revised_text = '\n'.join(revised_list)

    return revised_text