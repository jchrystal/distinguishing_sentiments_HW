

```python
import json
import pandas as pd
import tweepy
import numpy as np
from datetime import datetime as dt
import matplotlib.pyplot as plt
from config import consumer_key,consumer_secret,access_token,access_token_secret
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import time
analyzer = SentimentIntensityAnalyzer()
```


```python
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth, parser=tweepy.parsers.JSONParser())
```


```python
target = ['@BBC', '@CBS', '@CNN', '@FoxNews', '@NYTimes']
sent = []
for user in target:
    count = 0
    tweets = api.user_timeline(user, count=100)
    print(f'Analyzing {user}')
    for tweet in tweets:
        tweets_ago = count
        comp = analyzer.polarity_scores(tweet["text"])["compound"]
        pos = analyzer.polarity_scores(tweet["text"])["pos"]
        neg = analyzer.polarity_scores(tweet["text"])["neg"]
        neu = analyzer.polarity_scores(tweet["text"])["neu"]
        text = tweet["text"] 
        sent.append({"User": user,
                         "Date": tweet["created_at"],
                         "Compound": comp,
                         "Positive": pos,
                         "Negative": neg,
                         "Neutral": neu,
                         "Tweets Ago": count,
                         "Text": text})
        count = count + 1
```

    Analyzing @BBC
    Analyzing @CBS
    Analyzing @CNN
    Analyzing @FoxNews
    Analyzing @NYTimes
    


```python
sent_df = pd.DataFrame.from_dict(sent)
sent_df.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Compound</th>
      <th>Date</th>
      <th>Negative</th>
      <th>Neutral</th>
      <th>Positive</th>
      <th>Text</th>
      <th>Tweets Ago</th>
      <th>User</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0000</td>
      <td>Tue Mar 27 18:30:07 +0000 2018</td>
      <td>0.0</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>When mother Marie mysteriously leaves the fami...</td>
      <td>0</td>
      <td>@BBC</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.2942</td>
      <td>Tue Mar 27 18:00:08 +0000 2018</td>
      <td>0.0</td>
      <td>0.872</td>
      <td>0.128</td>
      <td>🇩🇪😂 Even if you don't speak German, this is wo...</td>
      <td>1</td>
      <td>@BBC</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.0000</td>
      <td>Tue Mar 27 17:00:07 +0000 2018</td>
      <td>0.0</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>🍜 We've got oodles of noodles with recipes for...</td>
      <td>2</td>
      <td>@BBC</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.0000</td>
      <td>Tue Mar 27 16:00:15 +0000 2018</td>
      <td>0.0</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>😬 What does Facebook know about you? https://t...</td>
      <td>3</td>
      <td>@BBC</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.6114</td>
      <td>Tue Mar 27 15:40:40 +0000 2018</td>
      <td>0.0</td>
      <td>0.750</td>
      <td>0.250</td>
      <td>RT @BBCTwo: Happy #WorldTheatreDay! *leaves th...</td>
      <td>4</td>
      <td>@BBC</td>
    </tr>
  </tbody>
</table>
</div>




```python
sent_df.to_csv("Twitter Sentiment Analysis.csv", index = False)
Date = dt.now().strftime("(%m/%d/%Y)")
```


```python
for user in target:
    user_df = sent_df.loc[sent_df["User"] == user]
    plt.scatter(user_df["Tweets Ago"],user_df["Compound"],label = user)
```


```python
plt.style.use('ggplot')
plt.title("Sentiment Analysis of Media Tweets "+str(Date))
plt.legend(bbox_to_anchor=(1,1), title='Media Sources')
plt.xlabel("Tweets Ago")
plt.ylabel("Tweet Polarity")
plt.savefig("Twitter Sentiment Analysis of News Outlets")
plt.show()
```


![png](output_6_0.png)



```python
group = sent_df.groupby("User")
means_sentiments = group["Compound"].mean()
means_sentiments.head()
```




    User
    @BBC        0.096663
    @CBS        0.358544
    @CNN       -0.045861
    @FoxNews   -0.007286
    @NYTimes   -0.033544
    Name: Compound, dtype: float64




```python
x_axis = np.arange(len(means_sentiments))
plt.bar(x_axis, means_sentiments, tick_label = target)
plt.title("Overall Media Sentiment based on Twitter " +str(Date))
plt.ylabel("Tweet Polarity")
plt.savefig("Overall Sentiment of News Tweets")
plt.show()
```


![png](output_8_0.png)

