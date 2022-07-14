import pandas as pd
import numpy as np
import json
import re 
import sys
import itertools
import warnings
from tabulate import tabulate
warnings.filterwarnings("ignore")
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from collections import defaultdict


class recommendation_system:
    
    def read_data(self , user_input, recommendation_number):
        """1. Data exploration"""
        #loading dataset
        idea_data = pd.read_csv('idea_data/idea_data.csv')

        #converting string format of Topics to list format
        idea_data['Topics'] = idea_data['Topics'].apply(lambda x: [re.sub(' ','_',i) for i in re.findall(r"'([^']*)'", x)])

        """2.Now lets do feature generation:
                a)Normalize float and int variables
                b)Bucketing of Popularity variable to create 10 classes of values ranging from 0-10.
                c)One-hot-encoding of Year and Popularity Variables
                d)Create TF-IDF features of strings in list of column Topics"""
        
        # creating 10 point buckets for popularity
        # Popularity is ranging from 0-100, so int division by 10 will create 10 different classes. 
        idea_data['Popularity_bucket'] = idea_data['Popularity'].apply(lambda x: int(x/10))
        feature_columns =['Year','Status','Popularity_bucket','school/institute/job']
        complete_feature_set_data = self.create_feature_set(idea_data, feature_columns)
        best_recommendations = self.recommend_songs(user_input,complete_feature_set_data,recommendation_number)
    
        return best_recommendations

    def one_hot_encoding(self,df, column, new_name): 
    
        tf_df = pd.get_dummies(df[column])
        feature_names = tf_df.columns
        tf_df.columns = [new_name + "|" + str(i) for i in feature_names]
        tf_df.reset_index(drop = True, inplace = True)    
        return tf_df

    def create_feature_set(self,df, int_float_feature_cols):
        """
            Process idea data to create a final set of features that will be used to generate recommendations

            Parameters: 
                df (pandas dataframe): Idea data
                int_float_feature_cols (list(str)): List of float columns that will be scaled 
            
            Returns: 
                final: final set of features 
        """
    
        #tfidf Topics lists
        tfidf = TfidfVectorizer()
        tfidf_matrix =  tfidf.fit_transform(df['Topics'].apply(lambda x: " ".join(x)))
        topics_feature = pd.DataFrame(tfidf_matrix.toarray())
        topics_feature.columns = ['topic' + "|" + i for i in tfidf.get_feature_names()]
        topics_feature.reset_index(drop = True, inplace=True)

        #explicity_ohe = ohe_prep(df, 'explicit','exp')    
        year_ohe = self.one_hot_encoding(df, 'Year','year') * 0.15 #scalling down the parameter of year, because it is not as important as topic features
        popularity_ohe = self.one_hot_encoding(df, 'Popularity_bucket','pop') * 0.25 #scalling down the parameter of Popularity bucket, because it is not as important as topic features

        #scale feature columns
        features = df[int_float_feature_cols].reset_index(drop = True)
        scaler = MinMaxScaler()
        feature_scaled = pd.DataFrame(scaler.fit_transform(features), columns = features.columns) * 0.2 #scalling down these features, because it is not as important as topic features

        #concanenate all features
        final = pd.concat([topics_feature, feature_scaled, popularity_ohe, year_ohe], axis = 1)
        
        #adding project name and its topics to feature dataframe, so that human readable outputs could be accessed
        final['Project_Name']=df['Project_Name'].values
        final['Topics']=df['Topics'].values
        
        return final
    
    def user_input_processing(self,user_input,complete_feature_set_data):
        
        user_input_features = complete_feature_set_data[(complete_feature_set_data['Project_Name'] == user_input)]
        user_input_features=user_input_features.drop(['Project_Name','Topics'], axis = 1)
        return user_input_features

    def recommend_songs(self,user_input,feature_idea_data,top_recommendations = 5):

        user_input_features = self.user_input_features_data = self.user_input_processing(user_input,feature_idea_data)

        #cosine similairity function:      
        feature_idea_data['similarity_ratio'] = cosine_similarity(feature_idea_data.drop(['Project_Name','Topics'], axis = 1).values, user_input_features.values[0].reshape(1, -1))[:,0]

        best_recommendations = feature_idea_data.sort_values('similarity_ratio',ascending = False)[['Project_Name','Topics']].iloc[1:top_recommendations+1]

        return best_recommendations

if __name__ == '__main__':
    recommendation_system_ = recommendation_system()
    user_typed_input = 'Multiporpose house-hold bot'
    output = recommendation_system_.read_data(user_typed_input,15)
    print(tabulate(output, headers='keys', tablefmt='grid', showindex=False))

    