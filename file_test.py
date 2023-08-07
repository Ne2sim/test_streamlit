#!/usr/bin/env python
# coding: utf-8

# In[35]:

#Packages
import pandas as pd
import numpy as np
import ipywidgets as widgets
import matplotlib.pyplot as plt
import warnings
import streamlit as st
from streamlit import components
import os
import matplotlib.font_manager as fm

hide_github_icon = """
<style>
.css-1jc7ptx, .e1ewe7hr3, .viewerBadge_container__1QSob, .styles_viewerBadge__1yB5_, .viewerBadge_link__1S137, .viewerBadge_text__1JaDK{ display: none; } #MainMenu{ visibility: hidden; } footer { visibility: hidden; } header { visibility: hidden; }
</style>
"""
st.markdown(hide_github_icon,unsafe_allow_html=True)

st.markdown("""
        <style>
               .block-container {
                    padding-top: 0rem;
                    padding-bottom: 0rem;
                    padding-right: 1rem;
                }
        </style>
        """, unsafe_allow_html=True)

#Remove Warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
pd.options.mode.chained_assignment = None 

st.markdown('<p style="font-size: 60px; font-weight: bold;">Role Ranking System</p>', unsafe_allow_html=True)
st.markdown('<p style="font-weight:bold; font-size: 20px; color: #808080;">By Chun Hang (<a href="https://twitter.com/chunhang7" target="_blank">Twitter</a> & <a href="https://www.instagram.com/chunhang7/" target="_blank">Instagram</a>): @chunhang7</p>', unsafe_allow_html=True)

with st.expander("App Details"):
    st.write('''
    The Role Ranking System assigns varrying weightages to different metrics based on their relevance to specific roles, reflecting the author's perspective backed by extensive research.\n 
    Similarity Function is based on K-Means Clustering, to identify similar players based on playstyle.\n
    
    Note: Only Outfielders from Top 7 Leagues with over 1080 Minutes Played in 2022/23 Season are Included for Selection.
    ''')

df = pd.read_csv("https://raw.githubusercontent.com/Lchunhang/StreamLit/main/FinalStats.csv")

df = df.loc[~(df['Position'] == 'Goalkeeper')]

#######################################################################

with st.sidebar:
    st.markdown('<h1 style="font-family: Consolas; font-size: 34px;">Select Your Players Here...</h1>', unsafe_allow_html=True)
    options = df["Player"].dropna().tolist()
    player = st.selectbox('Player', options)
    
    options2 = df["Position"].dropna().unique().tolist()
    default_position = df.loc[df['Player'] == player, 'Position'].values[0]
    user_input = st.selectbox("Enter Position Template (optional)", [default_position] + options2)

    if user_input != default_position:
        # Filter based on user input
        position = user_input
    else:
        # Default behavior with 'Position' derived from 'Player' column
        position = default_position

    # If the user has selected a position, update the position of the selected player
    if user_input != default_position:
        df.loc[df['Player'] == player, 'Position'] = position

    # Filter the dataframe based on the position
    df = df.loc[df['Position'] == position].reset_index(drop=True)
    st.write(position+" Template")
    st.markdown('<h1 style="font-family: Consolas; font-size: 34px;">..And Let The Magic Happen ➡️</h1>', unsafe_allow_html=True)

'''#Extract age, ready to merge
age = df[['Player','True Age', 'Minutes Played']]
age.rename(columns = {'True Age':'Age'},inplace = True)

# Note that `select_dtypes` returns a data frame. We are selecting only the columns
numeric_cols = df.select_dtypes(include=[np.number]).columns
df[numeric_cols] = df[numeric_cols].apply(zscore)

#Scale it to 100
x = df[numeric_cols]
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
x_scaled = x_scaled *100
df[numeric_cols] = pd.DataFrame(x_scaled)'''

#######################################################################

def CB_Rating(df):
    df = df

    return df 

#######################################################################

'''if not df.empty and 'Position' in df.columns and len(df['Position']) > 0:
    if df['Position'].iloc[0] == 'Centre-Back':
        df = CB_Rating(df)
        df = pd.merge(df, age, on="Player")
        df.rename(columns = {'Minutes Played_y':'Minutes Played'},inplace = True)
        
        df = df[['Player','True Position','Age','Squad','League']]

#######################################################################
df = df[['Player','True Position','Age','Squad']]'''

'''# Normalize the data using z-score scaling (standardization)
scaler = StandardScaler()
normalized_data = scaler.fit_transform(sdf[sdf.columns[7:]])

# Apply K-means clustering
num_clusters = 5 # You can choose the number of clusters based on your requirements
#kmeans = KMeans(n_clusters=num_clusters, n_init='auto', random_state=42)
kmeans = KMeans(n_clusters=num_clusters, n_init='auto')
clusters = kmeans.fit_predict(normalized_data)

# Add cluster labels to the DataFrame
sdf['Cluster'] = clusters

# Find the cluster of the selected player
selected_player_cluster = sdf.loc[sdf['Player'] == player, 'Cluster'].values[0]

# Get players in the same cluster as the selected player
similar_players = sdf[sdf['Cluster'] == selected_player_cluster]

# Check if the selected player exists in the cluster
if player in similar_players['Player'].values:
    
    # Extract the selected player's stats
    selected_player_stats = similar_players[similar_players['Player'] == player].iloc[0, 5:-1].values.astype(float)

 # Calculate the Euclidean distance of each player in the cluster from the selected player
    distances = np.linalg.norm(similar_players[similar_players.columns[5:-1]].values - selected_player_stats, axis=1)

    # Add the 'Distance' column to the DataFrame
    similar_players['Distance'] = distances

    # Sort the players by distance (similarity) and select the top 5 most similar players
    top_7_indices = distances.argsort()[1:8]  # Exclude the selected player itself
    top_7_similar_players = similar_players.iloc[top_7_indices]

top_7_similar_players = top_7_similar_players.iloc[:, :5].reset_index(drop=True)
top_7_similar_players = pd.merge(top_7_similar_players, age, on="Player")

top_7_similar_players = top_7_similar_players[['Player','Age_y','True Position','Squad','League','Minutes Played']]
top_7_similar_players.rename(columns = {'Age_y':'Age', 'True Position':'Main Position'},inplace = True)
top_7_similar_players.index += 1
Most_Similar = top_7_similar_players['Player'].values[0]
Age = top_7_similar_players['Age'].values[0]'''

#######################################################################

#Filter for player
'''df = df.loc[(df['Player'] == player)].reset_index(drop= True)

#add ranges to list of tuple pairs
values = []

for x in range(len(df['Player'])):
    if df['Player'][x] == player:
        values = df.iloc[x].values.tolist()
        

position = values[2]
age = values[3]
team = values[4]
minutes = values[6]
score1 = values[-3] 
score2 = values[-2]  
score3 = values[-1]  
values = values[7:13]

#get parameters
params = list(df.columns)
params = params[7:13]
params = [y[:-5] for y in params]

#get roles
roles = list(df.columns)
roles = roles[-3:]'''

#######################################################################

'''# color for the slices and text
slice_colors = ["#42b84a"] * 2 + ["#fbcf00"] * 2 + ["#39a7ab"] * 2
text_colors = ["#000000"] * 2 +  ["#000000"] * 2 + ["#000000"] * 2

# instantiate PyPizza class
baker = PyPizza(
    params=params,                    
    background_color="#f1e9d2",        
    straight_line_color="#000000",    
    straight_line_lw=2,               
    last_circle_color="black",      
    last_circle_lw= 5,                
    other_circle_lw=2,                
    inner_circle_size=0               
)

# plot pizza
fig, ax = baker.make_pizza(
    values,                          
    figsize=(8,10),                 
    color_blank_space="same",        
    slice_colors=slice_colors,       
    value_colors=text_colors,        
    value_bck_colors=slice_colors,   
    blank_alpha=0.4,                 
    kwargs_slices=dict(
        edgecolor="black", zorder=3, linewidth=4
    ),                              
    kwargs_params=dict(
        color="black", fontsize=17,fontfamily='DejaVu Sans',  va="center"
    ),                               
    kwargs_values=dict(
        color="black", fontsize=20,fontweight='bold',
         zorder=3,
        bbox=dict(
            edgecolor="black", facecolor="#FFFFFF",
            boxstyle="round,pad=0.2", lw=2.5
        )
    )                                
)


# add text
fig.text(
    1.4, 1.04, "Space",size=10, ha="center", fontweight='bold', color="none"
)

# add text
fig.text(
    1.4, 0.09, "Space",size=10, ha="center", fontweight='bold', color="none"
)

# add text
fig.text(
    0.095, 1, "Space",size=10, ha="center", fontweight='bold',color="none"
)

# add text
fig.text(
    0.75, 1.01, player + ", " + str(age) + " - " + team,size=32,
    ha="center", fontweight='bold',  color="black"
)

# add text
fig.text(
    0.75, 0.96, position + " Template | "+ str(minutes) + " Minutes Played",size=23,
    ha="center",  color="black"
)


# add text
fig.text(
    0.75, 0.91, "Most Similar Player: " + Most_Similar + ", "+ str(Age) ,size=20,
    ha="center", color="black"
)


fig.text(
    1.14, 0.78, score1, size=50,
    ha="left", fontweight='bold', color="black",
    bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3', lw=3)
)

# add text
fig.text(
    1.2, 0.68, roles[0] + "\nPercentile Rank" ,size=19,
    ha="center", fontweight='bold', color="black"
)

fig.text(
    1.14, 0.516, score2, size=50,
    ha="left", fontweight='bold', color="black",
    bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3', lw=3)
)

# add text
fig.text(
    1.2, 0.417, roles[1] + "\nPercentile Rank" ,size=19,
    ha="center", fontweight='bold', color="black"
)

fig.text(
    1.14, 0.267, score3, size=50,
    ha="left", fontweight='bold', color="black",
    bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3', lw=3)
)

# add text
fig.text(
    1.2, 0.166, roles[2] +"\nPercentile Rank" ,size=19,
    ha="center", fontweight='bold', color="black"
)

# add text
fig.text(
    0.75, 0.1, "Note: Top 7 European Leagues Players with 1080+ Minutes Included | Data: Opta | By @chunhang7" ,
    size=15, ha="center", color="black"
)

# Display the plot
st.pyplot(fig)'''

st.markdown('<p style="font-weight:bold; font-size: 20px; color: #808080;">Similar Players to ' + player + ' (' + position + ' Template)', unsafe_allow_html=True)
st.dataframe(df)

#with st.expander("What's Next"):
#    st.write('''
#    -> Comparing Feature\n
#    -> Goalkeeper Ranking System
#    ''')

#with st.expander("Special Thanks"):
#    st.write('''
#    Player Ratings was originally inspired by Scott Willis (@scottjwillis), Liam Henshaw (@HenshawAnalysis) & Andy Watson (@andywatsonsport).\n
#    Ben Griffis (@BeGriffis) was kind enough to share his previous work for me to draw Inspiration.\n
#    Joel A. Adejola (@joeladejola), Anuraag Kulkarni (@Anuraag027) , Rahul (@exceedingxpuns), Yusuf Raihan (@myusufraihan) & Daryl Dao (@dgouilard) for their thought-provoking review on the metrics applied here.\n
#    Thank you for the support!
#    ''')
