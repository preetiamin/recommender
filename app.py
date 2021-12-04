import streamlit as st
import pandas as pd
import numpy as np



HTML_WRAPPER = """<div style="overflow-x: auto; border: 1px solid #e6e9ef; border-radius: 0.25rem; padding: 1rem">{}</div>"""

st.title('Movie Recommendation System')
selected_method = st.radio('Rating Method', ('By User Rating', 'By Popularity'))

myurl = "https://liangfgithub.github.io/MovieData/ratings.dat"
ratings = pd.read_csv(myurl,encoding='ISO-8859-1',sep='::',header=None,engine='python')
ratings.columns=['UserID', 'MovieID', 'Rating', 'Timestamp']

myurl = "https://liangfgithub.github.io/MovieData/movies.dat"
movies = pd.read_csv(myurl,encoding='latin1',sep='::',header=None,engine='python')
movies.columns=['MovieID', 'Title', 'Genres']
movies['Year'] = movies['Title'].apply(lambda x:x[-5:-1])

genre_list = ["Action", "Adventure", "Animation", 
               "Children's", "Comedy", "Crime",
               "Documentary", "Drama", "Fantasy",
               "Film-Noir", "Horror", "Musical", 
               "Mystery", "Romance", "Sci-Fi", 
               "Thriller", "War", "Western"]

selected_genre = st.selectbox('Genre',genre_list,index=0)


for genre in genre_list:
    movies[genre] = movies['Genres'].apply(lambda x:1 if genre in x else 0)
    
averatings = ratings.groupby('MovieID')['Rating'].mean()
averatings.name = 'AveRating'
movies = movies.merge(averatings,on='MovieID')

numratings = ratings.groupby('MovieID')['Rating'].count()
numratings.name = 'NumRatings'
movies = movies.merge(numratings,on='MovieID')

def get_top_movies_by_rating(genre):
    st.write(genre)
    return movies[(movies['AveRating']>2.5) & (movies[genre]==1)].sort_values('AveRating', \
    ascending=False)[['MovieID','Title','AveRating']][:5]
 
def get_top_movies_by_popularity(genre):
    return movies[(movies['NumRatings']>1000) & (movies[genre]==1)].sort_values('NumRatings', \
    ascending=False)[['MovieID','Title','AveRating']][:5]


img_url = 'https://liangfgithub.github.io/MovieImages/'

if selected_method =='By User Rating':
    top_movies=get_top_movies_by_rating(selected_genre)
elif selected_method =='By Popularity':
    top_movies=get_top_movies_by_popularity(selected_genre)

st.write(genre)

st.write(top_movies)

cols = st.columns(5)
#st.write(img_url+'/'+str(top_movies.iloc[1,0])+'.jpg')
i=0
for col in cols:
    with col:
        st.image(img_url+'/'+str(top_movies.iloc[i,0])+'.jpg',width=100)
        st.write(top_movies.iloc[i,1][:-6])
        i+=1

