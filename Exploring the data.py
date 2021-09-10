#!/usr/bin/env python
# coding: utf-8

# In[28]:


# Import all the libraries required\n
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator


# In[3]:


S_DIR = r'geotagged_tweets_20160812-0912.tar\\geotagged_tweets_20160812-0912'
# with open(os.path.join(S_DIR, 'geotagged_tweets_20160812-0912.jsons')) as rf:
    #     data = json.load(rf)

df=pd.read_json('geotagged_tweets_20160812-0912.jsons', lines=True)


# In[4]:


print(type(df))


# In[ ]:


# pd.set_option('display.max_columns',None,'display.max_rows', None)
df.head()


# In[ ]:


df.tail()


# In[8]:


#https://www.datacamp.com/community/tutorials/wordcloud-python
print("There are {} observations and {} features in this dataset.".format(df.shape[0],df.shape[1]))


# In[13]:


print("There are {} languages in this dataset such as {}...".format(len(df.lang.unique()),", ".join(df.lang.unique()[0:5])))


# In[14]:


df.columns


# In[15]:


df.describe()


# In[16]:


print(df.coordinates.value_counts())


# In[20]:


#https://towardsdatascience.com/better-heatmaps-and-correlation-matrix-plots-in-python-41445d0f2bec
def heatmap(x, y, size):
    fig, ax = plt.subplots()
    
    # Mapping from column names to integer coordinates
    x_labels = [v for v in sorted(x.unique())]
    y_labels = [v for v in sorted(y.unique())]
    x_to_num = {p[1]:p[0] for p in enumerate(x_labels)} 
    y_to_num = {p[1]:p[0] for p in enumerate(y_labels)} 
    
    size_scale = 1000
    ax.scatter(
        x=x.map(x_to_num), # Use mapping for x
        y=y.map(y_to_num), # Use mapping for y
        s=size * size_scale, # Vector of square sizes, proportional to size parameter
        marker='s' # Use square as scatterplot marker
    )
    
    # Show column labels on the axes
    ax.set_xticks([x_to_num[v] for v in x_labels])
    ax.set_xticklabels(x_labels, rotation=45, horizontalalignment='right')
    ax.set_yticks([y_to_num[v] for v in y_labels])
    ax.set_yticklabels(y_labels)
    ax.grid(False, 'major')
    ax.grid(True, 'minor')
    ax.set_xticks([t + 0.5 for t in ax.get_xticks()], minor=True)
    ax.set_yticks([t + 0.5 for t in ax.get_yticks()], minor=True)
    
corr = df[df.columns].corr()
corr = pd.melt(corr.reset_index(), id_vars='index') # Unpivot the dataframe, so we can get pair of arrays for x and y
corr.columns = ['x', 'y', 'value']
heatmap(
    x=corr['x'],
    y=corr['y'],
    size=corr['value'].abs()
)


# In[21]:


#https://androidkt.com/plot-correlation-matrix-and-heatmaps-between-columns-using-pandas-and-seaborn/
import seaborn as sns
sns.heatmap(df.corr(), annot = True, fmt='.1g',cmap= 'coolwarm')


# In[23]:


len(df.columns)


# In[24]:


df.corr()


# In[26]:


#https://androidkt.com/plot-correlation-matrix-and-heatmaps-between-columns-using-pandas-and-seaborn/
# Correlated features, in general, donâ€™t improve models but they affect specific models in different ways and to varying extents. It is clear that correlated features means that they bring the same information, so it is logical to remove one of them.
upperMatrix = df.corr().where(np.triu(np.ones(df.corr().abs().shape), k=1).astype(np.bool))
# Find index of feature columns with correlation greater than 0.90
corrFutures = [column for column in upperMatrix.columns if any(upperMatrix[column] > 0.90)]
print(corrFutures)
# df.drop(columns=corrFutures)


# In[29]:


text=df['text']
print(text)


# In[32]:


# Groupby by language
language = df.groupby("lang")
# Summary statistic of all languages
language.describe().head()


# In[33]:


#https://www.datacamp.com/community/tutorials/wordcloud-python
plt.figure(figsize=(15,10))
language.size().sort_values(ascending=False).plot.bar()
plt.xticks(rotation=50)
plt.xlabel("Language")
plt.ylabel("Number of tweets")
plt.show()


# In[38]:


#https://canvas.uva.nl/courses/17514/files/folder/2020_Lectures?preview=3105392
#extract hyperlinks
def extract_hyperlink(text):
    regex=r'https?://[^\s<>"]+|www\.[^\s<>"]+'
    match=re.search(regex,text)
    if match:
        return match.group()
    return ''


# In[41]:


# Start with one tweet
text = df.text[5]
extract_hyperlink(text)

result = re.sub(r"http\S+","", text)
print(result)


# In[42]:


# Create and generate a word cloud image:
wordcloud = WordCloud(max_font_size=50, max_words=100, background_color="white").generate(text)
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()


# In[37]:


text_tweet = " ".join(tweet for tweet in df['text'])
print ("There are {} words in the combination of all tweets.".format(len(text_tweet)))


# In[45]:


text_tweet


# In[46]:


# Create stopword list:
stopwords = set(STOPWORDS)

#removing Links
result = re.sub(r"http\S+","", text_tweet)
    
# Generate a word cloud image
wordcloud = WordCloud(stopwords=stopwords, background_color="white").generate(result)

# Display the generated image:
# the matplotlib way:
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

