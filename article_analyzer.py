import en_core_web_lg
import pickle
import pandas as pd
from collections import Counter
from string import punctuation
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import json
from newsapi import NewsApiClient

# receive text and identify the Part-of-speech tagging
# to match keywords
def get_keywords_eng(str):
    result = []
    pos_tag = ['VERB', 'NOUN', 'PROPN']
    document = nlp_eng(str)
    for token in document:
        if (token.text in nlp_eng.Defaults.stop_words or token.text in punctuation):
            continue
        if (token.pos_ in pos_tag):
            result.append(token.text)

    return result


nlp_eng = en_core_web_lg.load()

# don't store creds in code
config_file = open("config.json")
config = json.load(config_file)

# nice try github api key scraping bots
newsapi = NewsApiClient(api_key=config['newsapi_key'])

articles = []

# 20 results per page, limit of 100 results
for pagina in range(1, 5):
    temp = newsapi.get_everything(q='coronavirus', language='en',
                                  from_param='2022-03-01', to='2022-03-23', sort_by='relevancy', page=pagina)
    # sanity check
    if not temp:
        break
    else:
        articles.append(temp)

pickle.dump(articles, open('articlesCOVID.pckl', 'wb'))

dados = []

# exctract data that we need
for i, article in enumerate(articles):
    for x in article['articles']:
        title = x['title']
        description = x['description']
        content = x['content']
        date = x['publishedAt']
        dados.append({'title': title, 'date': date,
                     'desc': description, 'content': content})

df = pd.DataFrame(dados)
df = df.dropna()
df.head()

results = []

# for each article, count the number of top 5 relevant words
for content in df.content.values:
    results.append([('#' + x[0])
                   for x in Counter(get_keywords_eng(content)).most_common(5)])

df['keywords'] = results

# save results
pickle.dump(df, open('articlesCOVID_analyzed.pckl', 'wb'))
df.to_csv(r'covid.csv', index=0)
df.head()

# generate word cloud
text = str(results)
wordcloud = WordCloud(max_font_size=50, max_words=100,
                      background_color="white").generate(text)
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()
