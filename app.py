# Core Pkgs
import streamlit as st 
import altair as alt
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
pio.templates.default = "plotly_white"

# EDA Pkgs
import pandas as pd 
import numpy as np
from datetime import datetime

# wordcloud & sns
from wordcloud import WordCloud
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import seaborn as sns
st.set_option('deprecation.showPyplotGlobalUse', False)
colours = ['#1F77B4', '#FF7F0E', '#2CA02C', '#DB2728', '#9467BD', '#8C564B', '#E377C2','#7F7F7F', '#BCBD22', '#17BECF','#67E568','#257F27','#08420D','#FFF000','#FFB62B','#E56124','#E53E30','#7F2353','#F911FF','#9F8CA6']
sns.set_palette(colours)
# %matplotlib inline
# import quantstats as qs
plt.rcParams['figure.figsize'] = (9, 6)
sns.set_style('darkgrid')

#NLP
from textprep import tweet_sentiment

# Utils
from tweets import api, get_tweet
import joblib
pipe_lr = joblib.load(open("modelnlp.pkl","rb"))

# Image
from PIL import Image

# Topic model
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import string

 
def get_timeline(username):
	timeline = api.user_timeline(
		# user_id=userID,
		screen_name=username,
		count=20,
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
	result = pd.DataFrame(list(zip(ids,at,text)),columns =['TweetID','Created','Tweet'])
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

emotions_emoji_dict = {"anger":"😠", "fear":"😨😱", "sad":"😔", "sadness":"😔", "shame":"😳","disgust":"🤮", "surprise":"😮","neutral":"😐", "happy":"🤗", "joy":"😂"}
color_discrete_map={"anger":"#de425b", "fear":"#e96678", "sad":"#f38794", "sadness":"#f38794", "shame":"#f7ada1","disgust":"#e8a985", "surprise":"#b2d7c8","neutral":"#ced4d0",
					"happy":"#70bda0", "joy":"#00a27a"}

sentiment_color={"Negative":"#e96678", "Neutral":"#ced4d0","Positive":"#70bda0"}

st.sidebar.title("Twitter User Insight and Behavior Observation")
usr = st.sidebar.text_input("Input Twitter User", value="")
# if st.sidebar.button("Analyze User"):
st.sidebar.write(f'Twitter Username: {usr}')
menu = ["Tweet Analyzer", "Topic Graph Analysis", "Recommendation"]
choice = st.sidebar.selectbox("Select Menu", menu)
data = getData(usr)
df = tweet_sentiment(data, 'Tweet')
if choice == "Tweet Analyzer":
	df['color']=df['Emotion'].map(color_discrete_map)
	col1,col2 = st.columns((1,1))
	with col1:
		df["EmoScore"] = df["EmoScore"] * df["EmoProba"]
		fig = px.pie(df, names='Emotion', values='EmoProba', color='Emotion', color_discrete_map=color_discrete_map)
		# fig = go.Figure(data=[go.Pie(labels=df['Emotion'], values=df['EmoProba'], hole=.4)])
		st.plotly_chart(fig)
		fig1 = px.scatter(df, x="Created", y="EmoScore",
						  color="Emotion", size='EmoProba', color_discrete_map=color_discrete_map)
		st.plotly_chart(fig1)

	with col2:
		df1 = df.groupby('Sentiment', as_index=False).agg({'Count': 'sum'})
		# fig0 = px.pie(df, names='Sentiment', values='Count', color='Sentiment', color_discrete_map=sentiment_color)
		fig0 = px.bar(df, x='Sentiment', y='Count', color='Sentiment', color_discrete_map=sentiment_color)
		st.plotly_chart(fig0)
		sw = stopwords.words('english')
		cek = df['Text_cleaned'].tolist()
		wordcloud = WordCloud(
			background_color='white',
			width=650,
			stopwords=set(
				sw + ['https', 'http', 'co', 'PT', 'the', 'and', 'for', 'a', 'an', 'to', 'from', 'am', 'is', 'has',
					  'have', 'had', 'do', 'did']),
			height=400
		).generate(' '.join(cek))
		wc = px.imshow(wordcloud)
		st.plotly_chart(wc)
		# fig2 = px.bar(df1, x='Emotion', y='EmoProba', color='Emotion', color_discrete_map=color_discrete_map)
		# st.plotly_chart(fig2)


	dff = df[['Tweet', 'Emotion','Sentiment','Emoji']]
	st.dataframe(dff)

elif choice == "Topic Graph Analysis":
	stop = set(stopwords.words('english'))
	exclude = set(string.punctuation)
	lemma = WordNetLemmatizer()

	def clean(doc):
		stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
		punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
		normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
		return normalized

	doc_clean = [clean(doc).split() for doc in df['Text_cleaned'].tolist()]
	import gensim
	from gensim import corpora

	# Creating the term dictionary of our courpus, where every unique term is assigned an index.
	dictionary = corpora.Dictionary(doc_clean)

	# Converting list of documents (corpus) into Document Term Matrix using dictionary prepared above.
	doc_term_matrix = [dictionary.doc2bow(doc) for doc in doc_clean]
	Lda = gensim.models.ldamodel.LdaModel
	ldamodel = Lda(doc_term_matrix, num_topics=3, id2word=dictionary, passes=50)
	topic = ldamodel.print_topics(num_topics=3, num_words=1)
	topic1 = topic[0][1]
	topic2 = topic[1][1]
	topic3 = topic[2][1]
	topic_list = [topic1.split('*')[1][1:-1], topic2.split('*')[1][1:-1], topic3.split('*')[1][1:-1]]
	st.write(topic_list)
	def extract_hashtags(text):
		hashtag_list = []
		for word in text.split():
			if word[0] == '#':
				hashtag_list.append(word[1:])
		return hashtag_list
	df1 = get_tweet(topic_list[0],50)
	# df2 = get_tweet(topic_list[1], 50)
	# df3 = get_tweet(topic_list[2], 50)
	# df1['hashtag'] = df1['Tweet'].apply(extract_hashtags)
	st.dataframe(df1)
	# st.dataframe(df2)
	# st.dataframe(df3)
	# doc1 = [clean(doc).split() for doc in df['Text_cleaned'].tolist()]
elif choice == "Recommendation":
	st.write("Channel to follow")
