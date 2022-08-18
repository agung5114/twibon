import tweepy
import pandas as pd
# api_key = st.secrets["api_key"]
# api_secret_key = st.secrets["api_secret_key"]
# access_token = st.secrets["access_token"]
# access_token_secret = st.secrets["access_token_secret"]
api_key = "l5FGlSMhD3FOB1phnwB7I9sX5"
api_secret_key = "R55gay8XG4uz1VGns8BT87zzXBGftNxPMaS9nvUVOzRI8YNsP1"
access_token = "237213820-RbW5PBW76TqcbT1tiAGjdkiMV7LlPnRIb9oDHixg"
access_token_secret = "EkHdik9UpmPB8CP8g3kSip0RC30LqgSRdkuGrovUnNEyN"
auth = tweepy.OAuthHandler(api_key,api_secret_key)
auth.set_access_token(access_token,access_token_secret)
api = tweepy.API(auth)

# Fxn
def get_tweet(kword,ntweet):
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