# Core Pkgs
import streamlit as st 
import altair as alt
import plotly.express as px 
import plotly.graph_objects as go

# EDA Pkgs
import pandas as pd 
import numpy as np 
from datetime import datetime
# import seaborn as sns
# st.set_option('deprecation.showPyplotGlobalUse', False)

#NLP
from textblob import TextBlob
from neattext import TextCleaner

# Utils
import joblib 
pipe_lr = joblib.load(open("modelnlp.pkl","rb"))
# pipe_ctm = joblib.load(open("model_custom.pkl","rb"))

import tweepy
api_key = st.secrets["api_key"]
api_secret_key = st.secrets["api_secret_key"]
access_token = st.secrets["access_token"]
access_token_secret = st.secrets["access_token_secret"]
auth = tweepy.OAuthHandler(api_key,api_secret_key)
auth.set_access_token(access_token,access_token_secret)
api = tweepy.API(auth)

# Image
from PIL import Image

# Fxn
def get_tweet(kword,ntweet):
    api = tweepy.API(auth, wait_on_rate_limit=True,wait_on_rate_limit_notify=True)
    search_hashtag = tweepy.Cursor(api.search_tweets, q=kword,result_type='popular', tweet_mode = "extended").items(ntweet)
    ids = []
    tweets = []
    users = []
    for tweet in search_hashtag:
        ids.append(tweet.user.id)
        users.append(tweet.user.name)
        tweets.append(tweet.full_text)
    result = pd.DataFrame(list(zip(ids,users,tweets)),columns =['ID','User', 'Tweet'])
    return result
 
def get_timeline(username):
	timeline = api.user_timeline(
		# user_id=userID,
		screen_name=username,
		count=10,
		include_rts = False,
		  # Necessary to keep full_text 
		  # otherwise only the first 140 words are extracted
		tweet_mode = 'extended'
	      )
	ids = []
	at = []
	text = []
	for info in timeline:
		ids.append(info.id)
		at.append(info.created_at)
		text.append(info.full_text)
		result = pd.DataFrame(list(zip(ids,at,text)),columns =['TweetID','Create_At','Tweet_Text'])
	return result

def predict_emotions(docx):
	results = pipe_lr.predict([docx])
	return results[0]

def get_prediction_proba(docx):
	results = pipe_lr.predict_proba([docx])
	return results

def predict_sentiment(docx):
    results = pipe_lr.predict([docx])
    return results[0]

def get_sentiment_proba(docx):
	results = pipe_lr.predict_proba([docx])
	return results

emotions_emoji_dict = {"anger":"😠","disgust":"🤮", "fear":"😨😱", "happy":"🤗", "joy":"😂", "neutral":"😐", "sad":"😔", "sadness":"😔", "shame":"😳", "surprise":"😮"}

# Main Application
def main():
    st.sidebar.title("Twitter Behavior Observation")
    usr = st.sidebar.text_input("Input Twitter User",value="")
    if st.sidebar.button("Analyze User"):
      st.sidebar.write(f'Twitter Username: {usr}')
    menu = ["Tweet Analyzer","Tweet Network","Recommendation"]
    choice = st.sidebar.selectbox("Select Menu", menu)
    if choice == "Tweet Analyzer":
        st.subheader("Tweet Analyzer")
	df = get_timeline(usr)
	search_text = df['Tweet'][0]
        #with st.form(key='emotion_clf_form'):
        #   search_text = st.text_area("Type Here")
        #   submit_text = st.form_submit_button(label='Submit')

        if submit_text:	
            hasilSearch = api.search_tweets(q=str(search_text),count=2)
            texts = []
            for tweet in hasilSearch:
                texts.append(tweet.text)
            # raw_text2 = texts[1]
            raw_text = texts[0]
            # translated = translator.translate(raw_text)
            # translated = raw_text
            prediction = predict_emotions(raw_text)
            probability = get_prediction_proba(raw_text)
            sentiment = predict_sentiment(raw_text)
            proba_sentiment = get_sentiment_proba(raw_text)
            col1,col2  = st.beta_columns(2)
            with col1:
                st.success("Search Result")
                st.write(raw_text)
                # st.write(raw_text2)
                # st.write(translated.text)
                st.success("Prediction")
                emoji_icon = emotions_emoji_dict[prediction]
                st.write("{}:{}".format(prediction,emoji_icon))
                st.write("Confidence:{}".format(np.max(probability)))
                st.write("Confidence:{}".format(np.max(proba_sentiment)))
            with col2:
                st.success("Prediction Probability")
                # st.write(probability)
                # st.write(proba_sentiment)
                proba_df = pd.DataFrame(probability,columns=pipe_lr.classes_)
                proba_sent_df = pd.DataFrame(proba_sentiment,columns=pipe_lr.classes_)
                # st.write(proba_df.T)
                # st.write(proba_sent_df.T)
                # proba_df_clean = proba_df.T.reset_index()
                # proba_df_clean.columns = ["emotions","probability"]
                proba_df_sent_clean = proba_sent_df.T.reset_index()
                proba_df_sent_clean.columns = ["sentiments","probability"]

                # fig = alt.Chart(proba_df_clean).mark_bar().encode(x='emotions',y='probability',color='emotions')
                # st.altair_chart(fig,use_container_width=True)
                fig = alt.Chart(proba_df_sent_clean).mark_bar().encode(x='sentiments',y='probability',color='sentiments')
                st.altair_chart(fig,use_container_width=True)
    elif choice == "Tweet Network":
        data = st.file_uploader("Upload Dataset", type=["csv","txt"])
        if data is not None:
            df = pd.read_csv(data)
            st.dataframe(df.head())
        else:
            st.write("No Dataset To Show")
        st.subheader("Exploratory Data Analysis")
        if data is not None:
            if st.checkbox("Show Shape"):
                st.write(df.shape)
            if st.checkbox("Show Summary"):
                st.write(df.describe())
            if st.checkbox("Correlation Matrix"):
                st.write(sns.heatmap(df.corr(),annot=True))
                st.pyplot()
    elif choice == "Recommendation":
        st.write("Channel to follow")

if __name__ == '__main__':
    main()
