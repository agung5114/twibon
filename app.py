# Core Pkgs
import streamlit as st 
import altair as alt
import plotly.express as px 
import plotly.graph_objects as go

# EDA Pkgs
import pandas as pd 
import numpy as np 
from datetime import datetime
import seaborn as sns
st.set_option('deprecation.showPyplotGlobalUse', False)

# Utils
import joblib 
pipe_lr = joblib.load(open("modelnlp.pkl","rb"))
pipe_ctm = joblib.load(open("model_custom.pkl","rb"))

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
def predict_emotions(docx):
	results = pipe_lr.predict([docx])
	return results[0]

def get_prediction_proba(docx):
	results = pipe_lr.predict_proba([docx])
	return results

def predict_sentiment(docx):
    results = pipe_ctm.predict([docx])
    return results[0]

def get_sentiment_proba(docx):
	results = pipe_ctm.predict_proba([docx])
	return results

emotions_emoji_dict = {"anger":"😠","disgust":"🤮", "fear":"😨😱", "happy":"🤗", "joy":"😂", "neutral":"😐", "sad":"😔", "sadness":"😔", "shame":"😳", "surprise":"😮"}
emoji_sentiment = {"positif":"🤗","negatif":"😔","netral":"😐","tdk-relevan":"😮"}

# Main Application
def main():
    st.title("Machine Learning Web Application")
    menu = ["Tweet Analyzer","Network","Recommendation"]
    choice = st.sidebar.selectbox("Select Menu", menu)
    search = st.sidebar.text_input("Input Twitter User",value="")
    if st.sidebar.button("Analyze User"):
      st.sidebar.write(f'Username  : {search}')
    if choice == "Sentiment":
        st.subheader("Tweet Analyzer")
        with st.form(key='emotion_clf_form'):
            search_text = st.text_area("Type Here")
            submit_text = st.form_submit_button(label='Submit')

        if submit_text:	
            hasilSearch = api.search(q=str(search_text),count=2)
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
                st.write("{}:{}".format(sentiment,emoji_sentiment[sentiment]))
                st.write("Confidence:{}".format(np.max(proba_sentiment)))
            with col2:
                st.success("Prediction Probability")
                # st.write(probability)
                # st.write(proba_sentiment)
                proba_df = pd.DataFrame(probability,columns=pipe_lr.classes_)
                proba_sent_df = pd.DataFrame(proba_sentiment,columns=pipe_ctm.classes_)
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
    elif choice == "Network":
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
    

if __name__ == '__main__':
    main()
