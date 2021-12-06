import streamlit as st
import pandas as pd
import numpy as np
from surprise import SVD
from surprise import Dataset
from surprise import Reader



HTML_WRAPPER = """<div style="overflow-x: auto; border: 1px solid #e6e9ef; border-radius: 0.25rem; padding: 1rem">{}</div>"""

st.title('Movie Recommendation System')

selected_type = st.sidebar.radio(
    "Recommendation Type",
    ("Top Movies by Genre", "Collaborative Filtering"))

genre_list = ["Action", "Adventure", "Animation", 
               "Children's", "Comedy", "Crime",
               "Documentary", "Drama", "Fantasy",
               "Film-Noir", "Horror", "Musical", 
               "Mystery", "Romance", "Sci-Fi", 
               "Thriller", "War", "Western"]

img_url = 'https://liangfgithub.github.io/MovieImages/'

@st.cache
def load_data():
    myurl = "https://liangfgithub.github.io/MovieData/ratings.dat"
    ratings = pd.read_csv(myurl,encoding='ISO-8859-1',sep='::',header=None,engine='python')
    ratings.columns=['UserID', 'MovieID', 'Rating', 'Timestamp']

    myurl = "https://liangfgithub.github.io/MovieData/movies.dat"
    movies = pd.read_csv(myurl,encoding='latin1',sep='::',header=None,engine='python')
    movies.columns=['MovieID', 'Title', 'Genres']
    movies['Year'] = movies['Title'].apply(lambda x:x[-5:-1])
    for genre in genre_list:
        movies[genre] = movies['Genres'].apply(lambda x:1 if genre in x else 0)

    averatings = ratings.groupby('MovieID')['Rating'].mean()
    averatings.name = 'AveRating'
    movies = movies.merge(averatings,on='MovieID')

    numratings = ratings.groupby('MovieID')['Rating'].count()
    numratings.name = 'NumRatings'
    movies = movies.merge(numratings,on='MovieID')
    return movies

movies = load_data()

def get_top_movies_by_rating(genre, n):
    return movies[(movies['NumRatings']>100) & (movies[genre]==1)].sort_values('AveRating', \
    ascending=False)[['MovieID','Title','AveRating']][:n]
 
def get_top_movies_by_popularity(genre, n):
    return movies[(movies['AveRating']>2) & (movies[genre]==1)].sort_values('NumRatings', \
    ascending=False)[['MovieID','Title','NumRatings']][:n]
  
def get_popular_movies(n):
    return movies.sort_values('NumRatings', ascending=False)[['MovieID','Title','AveRating']][:n]

def get_random_movies(n):
    return movies.sample(n)


if selected_type=='Top Movies by Genre':
                                                
    selected_method = st.sidebar.radio('Rating Method', ('By User Rating', 'By Popularity'))

    selected_genre = st.selectbox('Genre',genre_list,index=0)
    n=5
    
    if selected_method =='By User Rating':
        top_movies=get_top_movies_by_rating(selected_genre, n)
    elif selected_method =='By Popularity':
        top_movies=get_top_movies_by_popularity(selected_genre, n)

    rows = 1
    cols = st.columns(5)
    i=0
    for row in range(rows+1):
      for col in cols:
          with col:
              st.image(img_url+'/'+str(top_movies.iloc[i,0])+'.jpg',width=100)
              #st.write(top_movies.iloc[i,1][:-6])
              i+=1
              
elif selected_type=='Collaborative Filtering':
                                                
    selected_method = st.sidebar.radio('Rating Method', ('User Based', 'Item Based', 'SVD'))
    n=10

    user_item = ratings.pivot_table('Rating','UserID','MovieID')
    user_item.fillna(0,inplace=True)
    reader = Reader(rating_scale=(1, 5))

    data = Dataset.load_from_df(ratings[['UserID', 'MovieID', 'Rating']], reader)

    # Use the famous SVD algorithm.
    algo = SVD()

    st.caption('Rate some movies first, then click Submit to get recommendations')

    movies_to_rate = get_popular_movies(50)
    rows = 10
    
    with st.form(key='columns_in_form'):
        cols = st.columns(5)
        #st.write(img_url+'/'+str(top_movies.iloc[1,0])+'.jpg')
        i=0
        ratings = []
        for row in range(1,rows+1):
            for col in cols:
                with col:
                    st.image(img_url+'/'+str(movies_to_rate.iloc[i,0])+'.jpg',width=100)
                    val = col.selectbox(f'', ['Not Rated','1','2','3','4','5'], key=i)
                    user_item.loc[9999,movies_to_rate.iloc[i,0]]=val
                    i+=1
        submitted = st.form_submit_button('Submit')
        if submitted:
            st.write(user_item)
            #for i in range(rows*cols+1):

    '''
    if selected_method =='By User Rating':
        top_movies=get_top_movies_by_rating(selected_genre, n)
    elif selected_method =='By Popularity':
        top_movies=get_top_movies_by_popularity(selected_genre, n)
   '''








