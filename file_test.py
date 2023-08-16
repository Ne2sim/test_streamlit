#!/usr/bin/env python
# coding: utf-8

# In[35]:

#Packages
import pandas as pd
import numpy as np
#import ipywidgets as widgets
#import matplotlib.pyplot as plt
import warnings
import streamlit as st
#from streamlit import components
import os
import io
import xlsxwriter

# buffer to use for excel writer
buffer = io.BytesIO()
#import matplotlib.font_manager as fm

#hide_github_icon = """
#<style>
#.css-1jc7ptx, .e1ewe7hr3, .viewerBadge_container__1QSob, .styles_viewerBadge__1yB5_, .viewerBadge_link__1S137, .viewerBadge_text__1JaDK{ display: none; } #MainMenu{ visibility: hidden; } footer { visibility: hidden; } header { visibility: hidden; }
#</style>
#"""
#st.markdown(hide_github_icon,unsafe_allow_html=True)

#st.markdown("""
#		<style>
#			   .block-container {
#					padding-top: 0rem;
#					padding-bottom: 0rem;
#					padding-right: 1rem;
#				}
#		</style>
#		""", unsafe_allow_html=True)

#Remove Warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
pd.options.mode.chained_assignment = None 

st.set_page_config(layout="wide")

#st.markdown('<p style="font-size: 60px; font-weight: bold;">Role Ranking System</p>', unsafe_allow_html=True)
#st.markdown('<p style="font-weight:bold; font-size: 20px; color: #808080;">By Chun Hang (<a href="https://twitter.com/chunhang7" target="_blank">Twitter</a> & <a href="https://www.instagram.com/chunhang7/" target="_blank">Instagram</a>): @chunhang7</p>', unsafe_allow_html=True)

#with st.expander("App Details"):
#	st.write('''
#	The Role Ranking System assigns varrying weightages to different metrics based on their relevance to specific roles, reflecting the author's perspective backed by extensive research.\n 
#	Similarity Function is based on K-Means Clustering, to identify similar players based on playstyle.\n
#	
#	Note: Only Outfielders from Top 7 Leagues with over 1080 Minutes Played in 2022/23 Season are Included for Selection.
#	''')

if 1==1:
	c_base, c_plus = st.columns((1,1))

	with c_base:
		st.markdown('<h4 style="text-align:center;color: DeepSkyBlue;">Base file</h4>', unsafe_allow_html=True)

		df_base = pd.DataFrame()

		file_types = ['Excel','CSV and other text files']
		user_input_base = st.selectbox("File type", file_types, key='user_input_base')

		if user_input_base == 'CSV and other text files':
			#filepath_csv_base = st.text_input('Filepath', '/path/to/file', key='filepath_csv_base')
			filepath_csv_base = st.file_uploader("Upload a file", type=("csv","txt","tsv"), key='filepath_csv_base')

			c1_base, c2_base, c3_base, c4_base = st.columns(4)

			with c1_base:
				sep_base = st.text_input('Separator', ',', key='sep_base')
			with c2_base:
				header_types = ['Yes','No']
				header_csv_base = st.selectbox("First line as header", header_types, key='header_csv_base')
				if header_csv_base == 'Yes':
					header_csv_base = 0
				elif header_csv_base == 'No':
					header_csv_base = None
			with c3_base:
				quote_types = ['No quotes','Non-numeric only','All','Minimal']
				quoting_base = st.selectbox("Quoting", quote_types, key='quoting_base')
				if quoting_base == 'Minimal':
					quoting_base = 0
				elif quoting_base == 'All':
					quoting_base = 1
				elif quoting_base == 'Non-numeric only':
					quoting_base = 2
				elif quoting_base == 'No quotes':
					quoting_base = 3
				if quoting_base < 3:
					with c4_base:
						quotechar_base = st.text_input('Quote character', '', key='quotechar_base')
						try:
							df_base = pd.read_csv(filepath_csv_base,sep=sep_base,quoting=quoting_base,header=header_csv_base,quotechar=quotechar_base)
						except:
							pass
				else:
					try:
						df_base = pd.read_csv(filepath_csv_base,sep=sep_base,quoting=quoting_base,header=header_csv_base)
					except:
						pass


		elif user_input_base == 'Excel':
			#filepath = "/Users/nessim/Desktop/Projects/Streamlit/VlookUpOnline/data_test.csv"
			#filepath = "/Users/nessim/Desktop/Projects/Crypto Bot/3. Tracking/master_2021-09-29_20_37_11.xlsx"

			#filepath_xls_base = st.text_input('Filepath', '/path/to/file', key='filepath_xls_base')
			filepath_xls_base = st.file_uploader("Upload a file", type=("xlsx"), key='filepath_xls_base')
			
			#c1, c2, c3 = st.columns(3)

			#with c1:
			sheet_base = st.text_input('Sheet', 'Sheet1', key='sheet_base')
			#sheet_base = st.selectbox('Sheets',xl.sheet_names)
			#with c2:
			#	cell = st.text_input('Starting cell', 'A1:')
			#with c3:
			#	header_types = ['Yes','No']
			#	header_xls = st.selectbox("First line as header", header_types)
			#	if header_xls == 'No':
			#		userows -= 1

			#usecols = cell[:1]
			#userows = int(cell[1:])
			try:
				df_base = pd.read_excel(filepath_xls_base,sheet_name=sheet_base)
			except:
				pass	

		with st.expander("Preview"):
			if len(df_base)>0:
				st.dataframe(df_base,use_container_width=True)


	with c_plus:
		st.markdown('<h4 style="text-align:center;color: DeepSkyBlue;">Enrichment file</h4>', unsafe_allow_html=True)

		df_plus = pd.DataFrame()


		#file_types = ['Excel','CSV']
		user_input_plus = st.selectbox("File type", file_types, key='user_input_plus')

		if user_input_plus == 'CSV and other text files':
			#filepath_csv_plus = st.text_input('Filepath', '/path/to/file', key='filepath_csv_plus')
			filepath_csv_plus = st.file_uploader("Upload a file", type=("csv","txt","tsv"), key='filepath_csv_plus')
			c1_plus, c2_plus, c3_plus, c4_plus = st.columns(4)

			with c1_plus:
				sep_plus = st.text_input('Separator', ',', key='sep_plus')
			with c2_plus:
				header_types = ['Yes','No']
				header_csv_plus = st.selectbox("First line as header", header_types, key='header_csv_plus')
				if header_csv_plus == 'Yes':
						header_csv_plus = 0
				elif header_csv_plus == 'No':
						header_csv_plus = None
			with c3_plus:
				quote_types = ['No quotes','Non-numeric only','All','Minimal']
				quoting_plus = st.selectbox("Quoting", quote_types, key='quoting_plus')
				if quoting_plus == 'Minimal':
						quoting_plus = 0
				elif quoting_plus == 'All':
						quoting_plus = 1
				elif quoting_plus == 'Non-numeric only':
						quoting_plus = 2
				elif quoting_plus == 'No quotes':
						quoting_plus = 3
				if quoting_plus < 3:
					with c4_plus:
						quotechar_plus = st.text_input('Quote character', '', key='quotechar_plus')
						try:
							df_plus = pd.read_csv(filepath_csv_plus,sep=sep_plus,quoting=quoting_plus,header=header_csv_plus,quotechar=quotechar_plus)
						except:
							pass
				else:
					try:
						df_plus = pd.read_csv(filepath_csv_plus,sep=sep_plus,quoting=quoting_plus,header=header_csv_plus)
					except:
						pass


		elif user_input_plus == 'Excel':
			#filepath = "/Users/nessim/Desktop/Projects/Streamlit/VlookUpOnline/data_test.csv"
			#filepath = "/Users/nessim/Desktop/Projects/Crypto Bot/3. Tracking/master_2021-09-29_20_37_11.xlsx"
			#filepath_xls_plus = st.text_input('Filepath', '/path/to/file', key='filepath_xls_plus')
			filepath_xls_plus = st.file_uploader("Upload a file", type=("xlsx"), key='filepath_xls_plus')
			#c1, c2, c3 = st.columns(3)

			#with c1:
			sheet_plus = st.text_input('Sheet', 'Sheet1', key='sheet_plus')
			#with c2:
			#	cell = st.text_input('Starting cell', 'A1:')
			#with c3:
			#	header_types = ['Yes','No']
			#	header_xls = st.selectbox("First line as header", header_types)
			#	if header_xls == 'No':
			#		userows -= 1

			#usecols = cell[:1]
			#userows = int(cell[1:])
			try:
				df_plus = pd.read_excel(filepath_xls_plus,sheet_name=sheet_plus)
			except:
				pass

	
		with st.expander("Preview"):
			if len(df_plus)>0:
				st.dataframe(df_plus,use_container_width=True)


if len(df_base)>0 and len(df_plus)>0:
	st.write(' ')
	st.write(' ')
	c_l0, c_m0, c_r0 = st.columns((2,3,2))
	with c_l0:
		st.write(' ')
	with c_m0:
		st.markdown('<h4 style="text-align:center;color: DeepSkyBlue;">Merge files</h4>', unsafe_allow_html=True)
	with c_r0:
		st.write(' ')

	keys_left,keys_right = [],[]
	columns_left,columns_right = df_base.columns.insert(0, ""),df_plus.columns.insert(0, "")

	c_left, c_middle, c_right = st.columns((2,1,2))

	with c_left:
		key1_left = st.selectbox("First column to match", columns_left, key='key1_left')
		if key1_left!='':
			columns_left = columns_left.drop(key1_left)
			keys_left = keys_left + [key1_left]
			key2_left = st.selectbox("Second column to match (optional)", columns_left, key='key2_left')
			if key2_left!='':
				columns_left = columns_left.drop(key2_left)
				keys_left = keys_left + [key2_left]
				key3_left = st.selectbox("Third column to match (optional)", columns_left, key='key3_left')
				if key3_left!='':
					columns_left = columns_left.drop(key3_left)
					keys_left = keys_left + [key3_left]
	with c_middle:
		c_l, c_m, c_r = st.columns((1,3,1))
		with c_l:
			st.write(' ')
		with c_m:
			st.write(' ')
			st.write(' ')
			st.write('''should match with''')
		with c_r:
			st.write(' ')
		if key1_left!='':
			st.write(' ')
			c_l1, c_m1, c_r1 = c_middle.columns((1,3,1))
			with c_l1:
				st.write(' ')
			with c_m1:
				st.write(' ')
				st.write(' ')
				st.write('''should match with''')
			with c_r1:
				st.write(' ')
			if key2_left!='':
				st.write(' ')
				c_l2, c_m2, c_r2 = c_middle.columns((1,3,1))
				with c_l2:
					st.write(' ')
				with c_m2:
					st.write(' ')
					st.write(' ')
					st.write('''should match with''')
				with c_r2:
					st.write(' ')
	with c_right:
		key1_right = st.selectbox("First column to match", columns_right, key='key1_right')
		if key1_right!='':
			columns_right = columns_right.drop(key1_right)
			keys_right = keys_right + [key1_right]
			key2_right = st.selectbox("Second column to match (optional)", columns_right, key='key2_right')
			if key2_right!='':
				columns_right = columns_right.drop(key2_right)
				keys_right = keys_right + [key2_right]
				key3_right = st.selectbox("Third column to match (optional)", columns_right, key='key3_right')
				if key3_right!='':
					columns_right = columns_right.drop(key3_right)
					keys_right = keys_right + [key3_right]

	if len(keys_left)>0 and len(keys_right)>0 and len(keys_left)==len(keys_right):
		st.write(' ')
		st.write(' ')
		c_l00, c_m00, c_r00 = st.columns((2,3,2))
		with c_l00:
			st.write(' ')
		with c_m00:
			st.markdown('<h4 style="text-align:center;color: DeepSkyBlue;">Output file</h4>', unsafe_allow_html=True)
		with c_r00:
			st.write(' ')

		df_base[key1_left] = df_base[key1_left].astype(str)
		if key2_left != '':
			df_base[key2_left] = df_base[key2_left].astype(str)
			if key3_left != '':
				df_base[key3_left] = df_base[key3_left].astype(str)

		df_plus[key1_right] = df_plus[key1_right].astype(str)
		if key2_right != '':
			df_plus[key2_right] = df_plus[key2_right].astype(str)
			if key3_right != '':
				df_plus[key3_right] = df_plus[key3_right].astype(str)
		
		df_merged = df_base.merge(df_plus, left_on=keys_left, right_on=keys_right, how='left', suffixes=("", "_plus"), indicator=False)

		with st.expander("Preview"):
				if len(df_merged)>0:
					st.dataframe(df_merged,use_container_width=True)

		with st.expander("Download"):
				st.markdown(f"""You need a token to download your file, if you don't have one you can [get it there]({'https://donate.stripe.com/14k9CGgDngrh9nG145'})""")
				st.markdown(f"""If this page rendered you a service, you can pay the amount you want to get the token starting from 0,50€ 🙏""")
				st.markdown(f"""The token is then usable for life, without date limitation ❤️""")
				st.markdown(f"""Be watchful, the token is written in the confirmation message after payment 👀""")
				with st.form("login_form"):
					token = st.text_input('If you already have a token, enter it below 👇')
					#password = st.text_input('Enter Your Password')
					submitted = st.form_submit_button("Submit")


				if submitted:
					if token == '144f8140a4e8ec37b9cde1812cfd4048dd28c2e4639d9c5f8f4fff8b24a9aac5':#config('SECRET_PASSWORD'):
						st.session_state['logged_in'] = True
						st.text('Your token is valid! Download links will appear below in a few seconds ⌛')
					else:
						st.text('Unknown or expired token')
						st.session_state['logged_in'] = False


				if 'logged_in' in st.session_state.keys():
					if st.session_state['logged_in']:
						with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
						# Write each dataframe to a different worksheet.
							df_merged.to_excel(writer, sheet_name='Sheet1', index=False)

							writer.close()

						st.download_button(
						   label="Download as Excel",
						   data=buffer,
						   file_name='df_merged.xlsx',
						   mime='application/vnd.ms-excel'
						)

						csv = df_merged.to_csv(index=False).encode('utf-8')

						st.download_button(
						   "Download as CSV",
						   csv,
						   "file.csv",
						   "text/csv",
						   key='download-csv'
						)

		#/Users/nessim/Downloads/Copie de Controles des interfaces - export du réalisé TSQ Modèle_25.07.2023VF.xlsx
		#/Users/nessim/Downloads/Indicateurs_personnes_jour-20230725105004.csv
		#st.download_button('Download CSV', text_contents)  # Defaults to 'text/plain'

		#with open('myfile.csv') as f:
		#		st.download_button('Download CSV', f)



#except:
	# Prevent the error from propagating into your Streamlit app.
#	pass

# #######################################################################

# '''with st.sidebar:
#	 st.markdown('<h1 style="font-family: Consolas; font-size: 34px;">Select Your Players Here...</h1>', unsafe_allow_html=True)
#	 options = df["Player"].dropna().tolist()
#	 player = st.selectbox('Player', options)
	
#	 options2 = df["Position"].dropna().unique().tolist()
#	 default_position = df.loc[df['Player'] == player, 'Position'].values[0]
#	 user_input = st.selectbox("Enter Position Template (optional)", [default_position] + options2)

#	 if user_input != default_position:
#		 # Filter based on user input
#		 position = user_input
#	 else:
#		 # Default behavior with 'Position' derived from 'Player' column
#		 position = default_position

#	 # If the user has selected a position, update the position of the selected player
#	 if user_input != default_position:
#		 df.loc[df['Player'] == player, 'Position'] = position

#	 # Filter the dataframe based on the position
#	 df = df.loc[df['Position'] == position].reset_index(drop=True)
#	 st.write(position+" Template")
#	 st.markdown('<h1 style="font-family: Consolas; font-size: 34px;">..And Let The Magic Happen ➡️</h1>', unsafe_allow_html=True)'''

# '''#Extract age, ready to merge
# age = df[['Player','True Age', 'Minutes Played']]
# age.rename(columns = {'True Age':'Age'},inplace = True)

# # Note that `select_dtypes` returns a data frame. We are selecting only the columns
# numeric_cols = df.select_dtypes(include=[np.number]).columns
# df[numeric_cols] = df[numeric_cols].apply(zscore)

# #Scale it to 100
# x = df[numeric_cols]
# min_max_scaler = preprocessing.MinMaxScaler()
# x_scaled = min_max_scaler.fit_transform(x)
# x_scaled = x_scaled *100
# df[numeric_cols] = pd.DataFrame(x_scaled)'''

# #######################################################################

# '''def CB_Rating(df):
#	 df = df

#	 return df '''

# #######################################################################

# '''if not df.empty and 'Position' in df.columns and len(df['Position']) > 0:
#	 if df['Position'].iloc[0] == 'Centre-Back':
#		 df = CB_Rating(df)
#		 df = pd.merge(df, age, on="Player")
#		 df.rename(columns = {'Minutes Played_y':'Minutes Played'},inplace = True)
		
#		 df = df[['Player','True Position','Age','Squad','League']]

# #######################################################################
# df = df[['Player','True Position','Age','Squad']]'''

# '''# Normalize the data using z-score scaling (standardization)
# scaler = StandardScaler()
# normalized_data = scaler.fit_transform(sdf[sdf.columns[7:]])

# # Apply K-means clustering
# num_clusters = 5 # You can choose the number of clusters based on your requirements
# #kmeans = KMeans(n_clusters=num_clusters, n_init='auto', random_state=42)
# kmeans = KMeans(n_clusters=num_clusters, n_init='auto')
# clusters = kmeans.fit_predict(normalized_data)

# # Add cluster labels to the DataFrame
# sdf['Cluster'] = clusters

# # Find the cluster of the selected player
# selected_player_cluster = sdf.loc[sdf['Player'] == player, 'Cluster'].values[0]

# # Get players in the same cluster as the selected player
# similar_players = sdf[sdf['Cluster'] == selected_player_cluster]

# # Check if the selected player exists in the cluster
# if player in similar_players['Player'].values:
	
#	 # Extract the selected player's stats
#	 selected_player_stats = similar_players[similar_players['Player'] == player].iloc[0, 5:-1].values.astype(float)

#  # Calculate the Euclidean distance of each player in the cluster from the selected player
#	 distances = np.linalg.norm(similar_players[similar_players.columns[5:-1]].values - selected_player_stats, axis=1)

#	 # Add the 'Distance' column to the DataFrame
#	 similar_players['Distance'] = distances

#	 # Sort the players by distance (similarity) and select the top 5 most similar players
#	 top_7_indices = distances.argsort()[1:8]  # Exclude the selected player itself
#	 top_7_similar_players = similar_players.iloc[top_7_indices]

# top_7_similar_players = top_7_similar_players.iloc[:, :5].reset_index(drop=True)
# top_7_similar_players = pd.merge(top_7_similar_players, age, on="Player")

# top_7_similar_players = top_7_similar_players[['Player','Age_y','True Position','Squad','League','Minutes Played']]
# top_7_similar_players.rename(columns = {'Age_y':'Age', 'True Position':'Main Position'},inplace = True)
# top_7_similar_players.index += 1
# Most_Similar = top_7_similar_players['Player'].values[0]
# Age = top_7_similar_players['Age'].values[0]'''

# #######################################################################

# #Filter for player
# '''df = df.loc[(df['Player'] == player)].reset_index(drop= True)

# #add ranges to list of tuple pairs
# values = []

# for x in range(len(df['Player'])):
#	 if df['Player'][x] == player:
#		 values = df.iloc[x].values.tolist()
		

# position = values[2]
# age = values[3]
# team = values[4]
# minutes = values[6]
# score1 = values[-3] 
# score2 = values[-2]  
# score3 = values[-1]  
# values = values[7:13]

# #get parameters
# params = list(df.columns)
# params = params[7:13]
# params = [y[:-5] for y in params]

# #get roles
# roles = list(df.columns)
# roles = roles[-3:]'''

# #######################################################################

# '''# color for the slices and text
# slice_colors = ["#42b84a"] * 2 + ["#fbcf00"] * 2 + ["#39a7ab"] * 2
# text_colors = ["#000000"] * 2 +  ["#000000"] * 2 + ["#000000"] * 2

# # instantiate PyPizza class
# baker = PyPizza(
#	 params=params,					
#	 background_color="#f1e9d2",		
#	 straight_line_color="#000000",	
#	 straight_line_lw=2,			   
#	 last_circle_color="black",	  
#	 last_circle_lw= 5,				
#	 other_circle_lw=2,				
#	 inner_circle_size=0			   
# )

# # plot pizza
# fig, ax = baker.make_pizza(
#	 values,						  
#	 figsize=(8,10),				 
#	 color_blank_space="same",		
#	 slice_colors=slice_colors,	   
#	 value_colors=text_colors,		
#	 value_bck_colors=slice_colors,   
#	 blank_alpha=0.4,				 
#	 kwargs_slices=dict(
#		 edgecolor="black", zorder=3, linewidth=4
#	 ),							  
#	 kwargs_params=dict(
#		 color="black", fontsize=17,fontfamily='DejaVu Sans',  va="center"
#	 ),							   
#	 kwargs_values=dict(
#		 color="black", fontsize=20,fontweight='bold',
#		  zorder=3,
#		 bbox=dict(
#			 edgecolor="black", facecolor="#FFFFFF",
#			 boxstyle="round,pad=0.2", lw=2.5
#		 )
#	 )								
# )


# # add text
# fig.text(
#	 1.4, 1.04, "Space",size=10, ha="center", fontweight='bold', color="none"
# )

# # add text
# fig.text(
#	 1.4, 0.09, "Space",size=10, ha="center", fontweight='bold', color="none"
# )

# # add text
# fig.text(
#	 0.095, 1, "Space",size=10, ha="center", fontweight='bold',color="none"
# )

# # add text
# fig.text(
#	 0.75, 1.01, player + ", " + str(age) + " - " + team,size=32,
#	 ha="center", fontweight='bold',  color="black"
# )

# # add text
# fig.text(
#	 0.75, 0.96, position + " Template | "+ str(minutes) + " Minutes Played",size=23,
#	 ha="center",  color="black"
# )


# # add text
# fig.text(
#	 0.75, 0.91, "Most Similar Player: " + Most_Similar + ", "+ str(Age) ,size=20,
#	 ha="center", color="black"
# )


# fig.text(
#	 1.14, 0.78, score1, size=50,
#	 ha="left", fontweight='bold', color="black",
#	 bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3', lw=3)
# )

# # add text
# fig.text(
#	 1.2, 0.68, roles[0] + "\nPercentile Rank" ,size=19,
#	 ha="center", fontweight='bold', color="black"
# )

# fig.text(
#	 1.14, 0.516, score2, size=50,
#	 ha="left", fontweight='bold', color="black",
#	 bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3', lw=3)
# )

# # add text
# fig.text(
#	 1.2, 0.417, roles[1] + "\nPercentile Rank" ,size=19,
#	 ha="center", fontweight='bold', color="black"
# )

# fig.text(
#	 1.14, 0.267, score3, size=50,
#	 ha="left", fontweight='bold', color="black",
#	 bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3', lw=3)
# )

# # add text
# fig.text(
#	 1.2, 0.166, roles[2] +"\nPercentile Rank" ,size=19,
#	 ha="center", fontweight='bold', color="black"
# )

# # add text
# fig.text(
#	 0.75, 0.1, "Note: Top 7 European Leagues Players with 1080+ Minutes Included | Data: Opta | By @chunhang7" ,
#	 size=15, ha="center", color="black"
# )

# # Display the plot
# st.pyplot(fig)'''

# #st.markdown('<p style="font-weight:bold; font-size: 20px; color: #808080;">Similar Players to ' + player + ' (' + position + ' Template)', unsafe_allow_html=True)
# #st.dataframe(df)

# #with st.expander("What's Next"):
# #	st.write('''
# #	-> Comparing Feature\n
# #	-> Goalkeeper Ranking System
# #	''')

# #with st.expander("Special Thanks"):
# #	st.write('''
# #	Player Ratings was originally inspired by Scott Willis (@scottjwillis), Liam Henshaw (@HenshawAnalysis) & Andy Watson (@andywatsonsport).\n
# #	Ben Griffis (@BeGriffis) was kind enough to share his previous work for me to draw Inspiration.\n
# #	Joel A. Adejola (@joeladejola), Anuraag Kulkarni (@Anuraag027) , Rahul (@exceedingxpuns), Yusuf Raihan (@myusufraihan) & Daryl Dao (@dgouilard) for their thought-provoking review on the metrics applied here.\n
# #	Thank you for the support!
# #	''')
