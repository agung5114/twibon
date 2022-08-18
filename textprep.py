import pandas as pd
from textblob import TextBlob
from string import punctuation
import numpy as np

def remove_punctuations(text):
    # tuliskan algoritma remove punctuationnya
    for punct in punctuation:
        text = text.replace(punct, ' ')
    return text

import re

def deEmojify(text):
    regrex_pattern = re.compile(pattern = "["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags = re.UNICODE)
    return regrex_pattern.sub(r'',text)

def cleandf(df, text):
    # lowering text
    df['Text_cleaned'] = df[str(text)].str.lower()
    # remove tag: @
    df['Text_cleaned'] = df['Text_cleaned'].replace(to_replace=r'@(\w|\d)+', value=' ', regex=True)
    # remove tag: #
    df['Text_cleaned'] = df['Text_cleaned'].replace(to_replace=r'#(\w|\d)+', value=' ', regex=True)
    # remove link
    df['Text_cleaned'] = df['Text_cleaned'].replace(to_replace=r'(http|https)\S+', value=' ', regex=True)
    # remove number
    df['Text_cleaned'] = df['Text_cleaned'].replace(to_replace='\d+', value=' ', regex=True)
    # remove punctuations
    df['Text_cleaned'] = df['Text_cleaned'].apply(remove_punctuations)
    # remove emoji & spec character
    df['Text_cleaned'] = df['Text_cleaned'].apply(deEmojify)
    # remove unwanted spaces between words
    df['Text_cleaned'] = df['Text_cleaned'].replace(to_replace=' +', value=' ', regex=True)
    return df


def getPolarity(text):
    polarity = TextBlob(text).sentiment.polarity
    return polarity

import joblib
pipe_lr = joblib.load(open("modelnlp.pkl","rb"))
emotions_emoji_dict = {"anger":"😠", "fear":"😨😱", "sad":"😔", "sadness":"😔", "shame":"😳","disgust":"🤮", "surprise":"😮","neutral":"😐", "happy":"🤗", "joy":"😂"}
emotions_score_dict = {"anger":-2, "fear":-2, "sad":-1, "sadness":-1, "shame":-1,"disgust":-1, "surprise":0,"neutral":0, "happy":1, "joy":2}

def getEmo(docx):
    results = pipe_lr.predict([docx])
    return results[0]

def getEmoji(docx):
    results = pipe_lr.predict([docx])
    emoji_icon = emotions_emoji_dict[results[0]]
    return emoji_icon

def getEmoScore(docx):
    results = pipe_lr.predict([docx])
    emoji_score = emotions_score_dict[results[0]]
    return emoji_score

def getEmoProba(docx):
    results = pipe_lr.predict_proba([docx])
    confidence = np.max(results)
    return confidence

def tweet_sentiment(df, text):
    df = cleandf(df, text)
    df['Polarity'] = df['Text_cleaned'].apply(getPolarity)
    df['Count'] = 1
    df['Emotion'] = df[str(text)].apply(getEmo)
    df['Emoji'] = df[str(text)].apply(getEmoji)
    df['EmoScore'] = df[str(text)].apply(getEmoScore)
    df['EmoProba'] = df[str(text)].apply(getEmoProba)

    def getSentiment(score):
        if score < 0:
            return 'Negative'
        elif score == 0:
            return 'Neutral'
        else:
            return 'Positive'

    df['Sentiment'] = df['Polarity'].apply(getSentiment)
    return df
