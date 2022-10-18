# Core Pkgs
import streamlit as st

st.set_page_config(page_title="Twibon", page_icon=None, layout="wide", initial_sidebar_state="auto", menu_items=None)
# import altair as alt
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

pio.templates.default = "plotly_white"
from PIL import Image

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

# Utils
# from tweets import api, get_tweet, get_tags
from textprep import cleandf
# from st_aggrid import AgGrid
import snscrape.modules.twitter as sntwitter

# pipe_lr = joblib.load(open("modelnlp.pkl", "rb"))

# # Image
# from PIL import Image
            
def getTweetData(keyword,from_date,end_date,n):
    tweets_list2 = []
    # Using TwitterSearchScraper to scrape data and append tweets to list
    for i,tweet in enumerate(sntwitter.TwitterSearchScraper(f'{keyword} since:{from_date} until:{end_date}').get_items()):
        if i>(n-1):
            break
        tweets_list2.append([tweet.date, tweet.username,tweet.retweetCount, tweet.content])
        # tweet.retweetCount
    df = pd.DataFrame(tweets_list2, columns=['Datetime', 'Username','Retweeted','Text'])
    return df.sort_values(by='Retweeted',ascending=False)

color_map = {"Died": "#de425b", "Lost ": "#e96678", "Injured": "#f38794",
             "sadness": "#f38794", "shame": "#f7ada1",
                      "disgust": "#e8a985", "surprise": "#b2d7c8", "neutral": "#ced4d0",
                      "happy": "#70bda0", "joy": "#00a27a"}


import datetime
today = datetime.date.today()
# .strftime('%Y-%m-%d')
last30day = datetime.date.today() - datetime.timedelta(days=30)

@st.cache(allow_output_mutation=True)
def getFatalities():
    # kota = pd.read_csv('yearly.csv',sep=";")
    provinsi = pd.read_csv('series.csv',sep=";")
    provinsi = provinsi.fillna(0)
    df1 = provinsi.groupby(by=['Disaster','Province','Year'], as_index=False).agg({'Total': 'sum','Died':'sum','Lost ':'sum','Injured':'sum','Suffering':'sum','Evacuate':'sum'})
    return df1

dfplace = getFatalities()
dfplace['Province'] = dfplace['Province'].replace(to_replace=' ', value='', regex=True)
placelist = dfplace['Province'].str.lower().unique().tolist()

@st.cache(allow_output_mutation=True)
def getProperties():
    # kota = pd.read_csv('yearly.csv',sep=";")
    provinsi = pd.read_csv('series.csv',sep=";")
    provinsi = provinsi.fillna(0)
    df2 = provinsi.groupby(by=['Disaster','Province','Year'], as_index=False).agg({'Total': 'sum','House':'sum','School':'sum','Medical_Facility':'sum','Office':'sum','Factory':'sum','Bridge':'sum','Religious_Place':'sum'})
    return df2

# @st.cache(allow_output_mutation=True)
def getDisasters(disaster):
    data= getTweetData(disaster,last30day,today,100)
    # data['disaster'] = disaster
    dfc = cleandf(data,'Text')
    # cities = []
    return dfc

import streamlit.components.v1 as components

st.sidebar.image(Image.open('dews_transparent.png'))
st.sidebar.write(f'Disaster Early Warning System')
menu = ["Disaster Warning","Disaster Monitoring Dashboard","Disaster Search and Analysis"]
choice = st.sidebar.selectbox("Select Menu", menu)

if choice == "Disaster Warning":
    st.title("Disaster Early Warning From People's Tweets")
    dislist = ['banjir','gempa','longsor']
    disdict = {'banjir':'Flood','gempa':'Earthquake','longsor':'Landslides','tsunami':'Tsunami','erupsi':'Volcano_Eruption'}
    # st.write(placelist)
    dis1 = getDisasters(dislist[0])
    # dis1= getTweetData("banjir",,100)
    # # data['disaster'] = disaster
    # dis1 = cleandf(data,'Text')
    # st.dataframe(dis1)
    citylist1 =[]
    # placecount = 0
    for word in dis1['Text_cleaned'].tolist():
        # citylist =[]
        for place in placelist:
            if place in word.split():
                citylist1.append(place)
            else:
                pass
    dis2 = getDisasters(dislist[1])
    citylist2 =[]
    # placecount = 0
    for word in dis2['Text_cleaned'].tolist():
        # citylist =[]
        for place in placelist:
            if place in word.split():
                citylist2.append(place)
            else:
                pass
    dis3 = getDisasters(dislist[2])
    citylist3 =[]
    # placecount = 0
    for word in dis3['Text_cleaned'].tolist():
        # citylist =[]
        for place in placelist:
            if place in word.split():
                citylist3.append(place)
            else:
                pass

    k1,k2,k3 = st.columns((1,1,1))
    with k1:
        st.subheader("Banjir / Flood")
        Citycount = [1 for x in citylist1]
        fig1 = px.pie(names=citylist1, values=Citycount, color=citylist1, hole=.6)
        st.plotly_chart(fig1,use_container_width=True)
        st.dataframe(dis1[['Datetime','Text','Retweeted','Username']])
    with k2:
        st.subheader("Gempa / Earthquake")
        Citycount = [1 for x in citylist2]
        fig2 = px.pie(names=citylist2, values=Citycount, color=citylist2, hole=.6)
        st.plotly_chart(fig2,use_container_width=True)
        st.dataframe(dis2[['Datetime','Text','Retweeted','Username']])
    with k3:
        st.subheader("Tanah Longsor / LandSlides")
        Citycount = [1 for x in citylist3]
        fig3 = px.pie(names=citylist3, values=Citycount, color=citylist3, hole=.6)
        st.plotly_chart(fig3,use_container_width=True)
        st.dataframe(dis3[['Datetime','Text','Retweeted','Username']])
    
    # df = pd.concat([dis1,dis2,dis3])
    # # df['Date'] = pd.to_datetime(df['Datetime']).dt.date
    # df['Time']=df['Datetime'].dt.strftime('%H')
    # # fig4 = px.scatter(df,x='Date',y='Retweeted',size='Retweeted')
    # # fig4.update_layout(title_text="title", margin={"r": 0, "t": 40, "l": 0, "b": 0}, height=600)
    # # st.plotly_chart(fig4, use_container_width=True)
    # fig5 = px.scatter(df,x='Time',y='Retweeted',size='Retweeted')
    # fig5.update_layout(title_text="title", margin={"r": 0, "t": 40, "l": 0, "b": 0}, height=600)
    # st.plotly_chart(fig5, use_container_width=True)

elif "Disaster Monitoring Dashboard":
    components.html('''
        <div class='tableauPlaceholder' id='viz1666106127157' style='position: relative'><noscript><a href='#'><img alt='Disaster Monitoring Dashboard ' src='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;Di&#47;DisasterDashboard_16661046808440&#47;DisasterMonitoringDashboard&#47;1_rss.png' style='border: none' /></a></noscript><object class='tableauViz'  style='display:none;'><param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F' /> <param name='embed_code_version' value='3' /> <param name='site_root' value='' /><param name='name' value='DisasterDashboard_16661046808440&#47;DisasterMonitoringDashboard' /><param name='tabs' value='no' /><param name='toolbar' value='yes' /><param name='static_image' value='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;Di&#47;DisasterDashboard_16661046808440&#47;DisasterMonitoringDashboard&#47;1.png' /> <param name='animate_transition' value='yes' /><param name='display_static_image' value='yes' /><param name='display_spinner' value='yes' /><param name='display_overlay' value='yes' /><param name='display_count' value='yes' /><param name='language' value='en-US' /><param name='filter' value='publish=yes' /></object></div>                <script type='text/javascript'>                    var divElement = document.getElementById('viz1666106127157');                    var vizElement = divElement.getElementsByTagName('object')[0];                    if ( divElement.offsetWidth > 800 ) { vizElement.style.width='100%';vizElement.style.height=(divElement.offsetWidth*0.75)+'px';} else if ( divElement.offsetWidth > 500 ) { vizElement.style.width='100%';vizElement.style.height=(divElement.offsetWidth*0.75)+'px';} else { vizElement.style.width='100%';vizElement.style.height='1077px';}                     var scriptElement = document.createElement('script');                    scriptElement.src = 'https://public.tableau.com/javascripts/api/viz_v1.js';                    vizElement.parentNode.insertBefore(scriptElement, vizElement);                </script>
        ''',height=900,
            width=1440)

elif choice =="Disaster Search and Analysis":
    st.title("Realtime Search and Disaster Analysis From Tweets")
    k1,k2,k3,k4 = st.columns((1,1,1,1))
    with k1:
        search_term = st.text_input("Keyword")
    with k2:
        from_date = st.date_input("from date")
    with k3:
        end_date = st.date_input("until date")
    with k4:
        numb= st.number_input("Number to scrape",min_value=1,max_value=10000,step=100,value=100)
    
    # df = getFatalities()
    # df['Province'] = df['Province'].replace(to_replace=' ', value='', regex=True)
    # placelist = df['Province'].str.lower().unique().tolist()
    # st.write(placelist)
    if st.button("Run Scraping"):
        data= getTweetData(search_term,from_date,end_date,numb)
        # st.write("Latest Tweets")
        dfc = cleandf(data,'Text')
        # cities = []
        citylist =[]
        # placecount = 0
        for word in dfc['Text_cleaned'].tolist():
            # citylist =[]
            for place in placelist:
                if place in word.split():
                    citylist.append(place)
                else:
                    pass

        k1,k2 = st.columns((2,3))
        with k1:
            Citycount = [1 for x in citylist]
            fig1 = px.pie(names=citylist, values=Citycount, color=citylist, hole=.6)
            st.plotly_chart(fig1)
        with k2:
            st.dataframe(dfc[['Datetime','Text','Retweeted','Username']])
        # st.write(citylist)


# elif choice == "Disaster Stats":
#     # st.subheader("Disaster Records")
#     df1 = getFatalities()
#     df2 = getProperties()
#     c1, c2 = st.columns((1, 1))
#     with c1:
#         df1 = df1
#         fig1 = px.pie(df1, names='Disaster', values='Total', color='Disaster', hole=.6)
#         st.plotly_chart(fig1)
        
#     with c2:
#         df2b = df2.groupby(by=['Disaster','Province'],as_index=False).agg({'Total':'sum'})
#         fig2 = px.bar(df2b, x='Province', y='Total', color='Disaster')
#         fig2.update_xaxes(title_text="")
#         st.plotly_chart(fig2)

#     c3, c4 = st.columns((2, 2))
#     with c3:
#         trace1 = go.Bar(
#                     x = df1['Province'],
#                     y = df1['Died'],
#                     marker=dict(color="#de425b"),
#                     name = 'Total Died'
#                     )
#         trace2 = go.Bar(
#                     x = df1['Province'],
#                     y = df1['Lost '],
#                     name = 'Total Lost',
#                     marker=dict(color="#e8a985")
#                     )
#         trace3 = go.Bar(
#                     x = df1['Province'],
#                     y = df1['Injured'],
#                     name = 'Total Injured',
#                     marker=dict(color = '#c0c2a9')
#                     )
#         data = [trace3, trace2, trace1]
#         layout = go.Layout(barmode = 'stack')
#         fig3 = go.Figure(data = data, layout = layout)
#         fig3.update_layout(title ="Total Fatalities")
#         # fig1 = px.bar(df1, x='Province', y='Total', color='Disaster')
#         st.plotly_chart(fig3)
#     with c4:
#         # AgGrid(df1,fit_columns_on_grid_load=True)
#         st.write("Detailed Data")
#         for c in df1:
#             if df1[c].dtype == np.float:
#                 df1[c] = df1[c].astype(int)
#         st.dataframe(df1)
#     st.components.v1.html("<a class="twitter-timeline" href="https://twitter.com/HarvardHealth?ref_src=twsrc%5Etfw">Tweets by HarvardHealth</a> <script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>",width=None, height=None, scrolling=False)
