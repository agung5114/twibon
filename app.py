# Core Pkgs
import streamlit as st

st.set_page_config(page_title="Twibon", page_icon=None, layout="wide", initial_sidebar_state="auto", menu_items=None)
# import altair as alt
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

pio.templates.default = "plotly_white"

import subprocess
cmd = ['python3','-m','textblob.download_corpora']
subprocess.run(cmd)

# EDA Pkgs
import pandas as pd
import numpy as np
from datetime import datetime

# wordcloud & sns
from wordcloud import WordCloud
import warnings

warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
# import seaborn as sns

st.set_option('deprecation.showPyplotGlobalUse', False)
# colours = ['#1F77B4', '#FF7F0E', '#2CA02C', '#DB2728', '#9467BD', '#8C564B', '#E377C2', '#7F7F7F', '#BCBD22', '#17BECF',
#            '#67E568', '#257F27', '#08420D', '#FFF000', '#FFB62B', '#E56124', '#E53E30', '#7F2353', '#F911FF', '#9F8CA6']
# sns.set_palette(colours)
# # %matplotlib inline
# # import quantstats as qs
# plt.rcParams['figure.figsize'] = (9, 6)
# sns.set_style('darkgrid')

# Utils
from tweets import api, get_tweet
from textprep import tweet_sentiment, cleandf
from network import networkFig
import joblib

pipe_lr = joblib.load(open("modelnlp.pkl", "rb"))

# Image
from PIL import Image

# Topic model
# from nltk.corpus import stopwords
# from nltk.stem.wordnet import WordNetLemmatizer
# import string
# import nltk
# nltk.download('punkt')


def get_timeline(username):
    timeline = api.user_timeline(
        # user_id=userID,
        screen_name=username,
        count=20,
        include_rts=False,
        # Necessary to keep full_text
        # otherwise only the first 140 words are extracted
        tweet_mode='extended'
    )
    ids = []
    at = []
    text = []
    for info in timeline:
        ids.append(info.id)
        at.append(info.created_at)
        text.append(info.full_text)
    result = pd.DataFrame(list(zip(ids, at, text)), columns=['TweetID', 'Created', 'Tweet'])
    return result
#
def predict_emotions(docx):
    results = pipe_lr.predict([docx])
    return results[0]


def get_prediction_proba(docx):
    results = pipe_lr.predict_proba([docx])
    return results


@st.cache(allow_output_mutation=True)
def getData(usr):
    df = get_timeline(usr)
    return df


from textblob import TextBlob

stopw = pd.read_csv('stop.csv')
stop = set(['i','m','u']+stopw['stopword'].tolist())
emotions_emoji_dict = {"anger": "😠", "fear": "😨😱", "sad": "😔", "sadness": "😔", "shame": "😳", "disgust": "🤮",
                       "surprise": "😮", "neutral": "😐", "happy": "🤗", "joy": "😂"}
color_discrete_map = {"anger": "#de425b", "fear": "#e96678", "sad": "#f38794", "sadness": "#f38794", "shame": "#f7ada1",
                      "disgust": "#e8a985", "surprise": "#b2d7c8", "neutral": "#ced4d0",
                      "happy": "#70bda0", "joy": "#00a27a"}

sentiment_color = {"Negative": "#e96678", "Neutral": "#ced4d0", "Positive": "#70bda0"}

st.sidebar.title("Twitter User Insight and Behavior Observation")
usr = st.sidebar.text_input("Input Twitter User", value="morgwenmagic")
# if st.sidebar.button("Analyze User"):
st.sidebar.write(f'Twitter Username: {usr}')
menu = ["User's Tweet Analysis", "Topic Graph Analysis", "Recommendation"]
choice = st.sidebar.selectbox("Select Menu", menu)
data = getData(usr)
df = tweet_sentiment(data, 'Tweet')
if choice == "User's Tweet Analysis":
    st.subheader("User's Behavior and Tweet Activities")
    df['color'] = df['Emotion'].map(color_discrete_map)
    col1, col2 = st.columns((1, 1))
    with col1:
        df["EmoScore"] = df["EmoScore"] * df["EmoProba"]
        fig = px.pie(df, names='Emotion', values='EmoProba', color='Emotion', color_discrete_map=color_discrete_map)
        # fig = go.Figure(data=[go.Pie(labels=df['Emotion'], values=df['EmoProba'], hole=.4)])
        st.plotly_chart(fig)
#         fig1 = px.scatter(df, x="Created", y="EmoScore",
#                           color="Emotion", size='EmoProba', color_discrete_map=color_discrete_map)
#         st.plotly_chart(fig1)
        
        dff = df[['Created', 'Tweet', 'Emotion', 'Sentiment', 'Emoji']]
        st.dataframe(dff)
#         blob = TextBlob(" ".join(i for i in df['Text_cleaned'].tolist()))
#         verbs = list()
#         for word, tag in blob.tags:
#             if tag == 'VB':
#                 verbs.append(word.lemmatize())
#         wordcloud1 = WordCloud(
#             background_color='white',
#             width=650,
#             stopwords=stop,
#             height=400
#         ).generate(' '.join(verbs))
#         wc1 = px.imshow(wordcloud1, title='Verb WordCloud')
#         st.plotly_chart(wc1)

    with col2:
        # df1 = df.groupby('Sentiment', as_index=False).agg({'Count': 'sum'})
#         df1 = df.groupby('Emotion', as_index=False).agg({'EmoProba': 'sum'})
#         # fig0 = px.pie(df, names='Sentiment', values='Count', color='Sentiment', color_discrete_map=sentiment_color)
#         # fig0 = px.bar(df, x='Sentiment', y='Count', color='Sentiment', color_discrete_map=sentiment_color)
#         fig0 = px.bar(df1, x='Emotion', y='EmoProba', color='Emotion', color_discrete_map=color_discrete_map)
#         st.plotly_chart(fig0)
        fig1 = px.scatter(df, x="Created", y="EmoScore",
                          color="Emotion", size='EmoProba', color_discrete_map=color_discrete_map)
        st.plotly_chart(fig1)
        blob = TextBlob(" ".join(i for i in df['Text_cleaned'].tolist()))
        cek = []
        for nouns in blob.noun_phrases:
            cek.append(nouns)
        wordcloud2 = WordCloud(
            background_color='white',
            width=650,
            stopwords=stop,
            height=400
        ).generate(' '.join(cek))
        wc2 = px.imshow(wordcloud2, title='Noun WordCloud')
        st.plotly_chart(wc2)
    # fig2 = px.bar(df1, x='Emotion', y='EmoProba', color='Emotion', color_discrete_map=color_discrete_map)
    # st.plotly_chart(fig2)

elif choice == "Topic Graph Analysis":
    st.subheader("Topic Related to User's Tweets and Influencer Network")
    doc_clean = []
    for sentence in df['Text_cleaned']:
        stop_free = " ".join([i for i in sentence.lower().split() if i not in stop])
        doc_clean.append(stop_free)

    # Extract noun
    from textblob import TextBlob
    blob = TextBlob(" ".join(i for i in df['Text_cleaned'].tolist()))
    nn_count = []
    topic = []
    for nouns in blob.noun_phrases:
        if nouns not in stop:
            nn_count.append(blob.noun_phrases.count(nouns))
            topic.append(nouns)
            df_topic = pd.DataFrame(list(zip(topic,nn_count)),columns =['Topic','Freq'])
    df_topic = df_topic.sort_values(by='Freq', ascending=False).drop_duplicates()
    # st.write(df_topic)

    try:
        df1 = cleandf(get_tweet(df_topic['Topic'][0], 15), 'Tweet')
        result1 = api.retweets(df1['ID'][0])
        result2 = api.retweets(df1['ID'][1])
        result3 = api.retweets(df1['ID'][2])
    except:
        df1 = cleandf(get_tweet(df_topic['Topic'][1], 15), 'Tweet')
        result1 = api.retweets(df1['ID'][0])
        result2 = api.retweets(df1['ID'][1])
        result3 = api.retweets(df1['ID'][2])

    target = []
    source = []
    for i in range(len(result1)):
        usr = result1[i]._json['user']['screen_name']
        target.append(usr)
        source.append(df1['User'][0])

    for i in range(len(result2)):
        usr = result2[i]._json['user']['screen_name']
        target.append(usr)
        source.append(df1['User'][1])

    for i in range(len(result3)):
        usr = result3[i]._json['user']['screen_name']
        target.append(usr)
        source.append(df1['User'][2])

    dfnet = pd.DataFrame(list(zip(source,target)), columns=['Source', 'Target'])
    # st.dataframe(dfnet)
    netfig = networkFig(dfnet,'Source','Target','Network Graph Analysis')
    k1,k2 = st.columns((1,1))
    with k1:
        st.dataframe(df1[['User', 'Retweeted', 'Tweet']])
    with k2:
        st.plotly_chart(netfig)

elif choice == "Recommendation":
    st.write("Channel to follow")
