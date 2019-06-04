import pandas as pd
import time
import numpy as np
from rake_nltk import Rake
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer


# Read the csv file
df = pd.read_csv('../datasets/tv-shows-database.csv')

# Choose what columns to show
df = df[['Title','Actors', 'Writer', 'Genre', 'Plot']]
# df = df[['Title', 'Genre', 'Plot']]

df.head()

# putting the genres in a list of words
df['Genre'] = df['Genre'].map(lambda x: x.lower().split(','))

df['Actors'] = df['Actors'].map(lambda x: x.split(','))

df['Writer'] = df['Writer'].map(lambda x: x.split(' '))

# merging together first and last name for each actor and director, so it's considered as one word
# and there is no mix up between people sharing a first name
for index, row in df.iterrows():
    row['Actors'] = [x.lower().replace(' ','') for x in row['Actors']]
    row['Writer'] = ''.join(row['Writer']).lower()

    # initializing the new column
    df['Key_words'] = ""

    for index, row in df.iterrows():
        plot = row['Plot']

        # instantiating Rake, by default is uses english stopwords from NLTK
        # and discard all puntuation characters
        r = Rake()

        # extracting the words by passing the text
        r.extract_keywords_from_text(plot)

        # getting the dictionary whith key words and their scores
        key_words_dict_scores = r.get_word_degrees()

        # assigning the key words to the new column
        row['Key_words'] = list(key_words_dict_scores.keys())

    # dropping the Plot column
    # df.drop(columns = ['Plot'], inplace = True)

df.set_index('Title', inplace = True)
df.head()

df['bag_of_words'] = ''
columns = df.columns
for index, row in df.iterrows():
    words = ''
    for col in columns:
        if col != 'Writer':
            words = words + ' '.join(row[col])+ ' '
        else:
            words = words + row[col]+ ' '
    row['bag_of_words'] = words

df.drop(columns = [col for col in df.columns if col!= 'bag_of_words'], inplace = True)

# instantiating and generating the count matrix
count = CountVectorizer()
count_matrix = count.fit_transform(df['bag_of_words'])

# creating a Series for the movie titles so they are associated to an ordered numerical
# list I will use later to match the indexes
indices = pd.Series(df.index)
indices[:5]

# generating the cosine similarity matrix
cosine_sim = cosine_similarity(count_matrix, count_matrix)
cosine_sim


# function that takes in movie title as input and returns the top 10 recommended movies
def recommendations(title, cosine_sim = cosine_sim):

    recommended_movies = []

    # gettin the index of the movie that matches the title
    idx = indices[indices == title].index[0]

    # creating a Series with the similarity scores in descending order
    score_series = pd.Series(cosine_sim[idx]).sort_values(ascending = False)

    # getting the indexes of the 10 most similar movies
    top_10_indexes = list(score_series.iloc[1:11].index)

    # populating the list with the titles of the best 10 matching movies
    print('Baseado no seu gosto por "{}" você deveria assistir: '.format(title))
    for i in top_10_indexes:
        recommended_movies.append(list(df.index)[i])

    for x in range(len(recommended_movies)):
        print('{0}: {1}'.format(x+1, recommended_movies[x]))

start_time = time.time()

print('----Filtragem de Conteúdo----\n')
recommendations('Criminal Minds')

print ('\nTotal Runtime: {:.2f} seconds'.format(time.time() - start_time))
