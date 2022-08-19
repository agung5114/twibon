import tweepy
import pandas as pd
import streamlit as st
api_key = st.secrets["api_key"]
api_secret_key = st.secrets["api_secret_key"]
access_token = st.secrets["access_token"]
access_token_secret = st.secrets["access_token_secret"]
auth = tweepy.OAuthHandler(api_key,api_secret_key)
auth.set_access_token(access_token,access_token_secret)
api = tweepy.API(auth)

# import snscrape.modules.twitter as sntwitter
# def getTweetData(keyword,n):
#     tweets_list2 = []
#     # Using TwitterSearchScraper to scrape data and append tweets to list
#     for i,tweet in enumerate(sntwitter.TwitterSearchScraper(f'{keyword} ').get_items()):
#         if i>10:
#             break
#         tweets_list2.append([tweet.date, tweet.id, tweet.username,tweet.retweetCount, tweet.content])
#         # tweet.retweetCount
#         df = pd.DataFrame(tweets_list2, columns=['Datetime', 'Tweet Id', 'Username','Retweeted','Tweet'])
#     return df.sort_values(by='Retweeted',ascending=False)
# Fxn
def get_tweet(kword,ntweet):
    search_hashtag = tweepy.Cursor(api.search, q=kword,result_type='popular', tweet_mode = "extended").items(ntweet)
    ids = []
    tweets = []
    users = []
    retweets = []
    for tweet in search_hashtag:
        # ids.append(tweet.user.id)
        ids.append(tweet.id)
        users.append(tweet.user.screen_name)
        tweets.append(tweet.full_text)
    for id in ids:
        status = api.get_status(id, tweet_mode="extended")
        retweets.append(status.retweet_count)
    result = pd.DataFrame(list(zip(ids,users,retweets,tweets)),columns =['ID','User','Retweeted','Tweet'])
    return result.sort_values(by='Retweeted', ascending=False)
