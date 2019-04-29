#We will use Scikit-Surprise for Recommender Systems
#We will implement one Content Based and one Colaborative Filter recommendation systems,
#and use both to create a hybrid model.

import pandas as pd
import numpy as np
from sklearn.metrics import pairwise
import datetime

class NewsRecommender:
	def __init__(self,trained_model=False):
		if not trained_model:
			self.news_df = pd.read_csv('wrangled_data/news_df.csv')
			self.analytics_df = pd.read_csv('wrangled_data/analytics_df.csv')
			self.position_dic = {}

	#Private function to calculate the content based ranking score
	def __calculate_content_based_rank(self,x,this_vector,this_date):
		cosine_rank = pairwise.cosine_similarity(x[0].reshape(1,-1),this_vector.reshape(1,-1))*1000
		date_rank = (5-int((this_date-x[1]).days))*100
		result = cosine_rank + date_rank
		#print(result[0])
		return result[0][0]

	#Private function to create a vector for a given news
	def __create_vector(self,x,position_dic):
		result_array = np.zeros(len(position_dic.keys()))
		tag_list = x.lower().replace(' ','').split(',')
		for tag in tag_list:
			result_array[position_dic[tag]]=1
		return result_array


	#Produces a vector matrix based on the tags of news
	def train_content_based(self):
		#Build vector dimensions
		tag_list = self.news_df['tags'].tolist()
		for tag in tag_list:
			tags = tag.lower().replace(' ','').split(',')
			for item in tags:
				if item not in self.position_dic.keys():
					self.position_dic[item] = len(self.position_dic.keys())

		#print(position_dic.keys())
		self.news_df['tag_array'] = self.news_df['tags'].apply(lambda x:self.__create_vector(x,self.position_dic))
		news_df_csv = self.news_df.copy()
		news_df_csv['tag_array'] = news_df_csv['tag_array'].apply(lambda x: x.tolist())
		news_df_csv.to_csv('state/news_state.csv')

	#Returns a list of n recomendations based on the current tags and date
	def recommend_content(self,n,tags,date):
		vector = self.__create_vector(tags,self.position_dic)
		self.news_df['date_object'] = self.news_df['publishedAt'].apply(lambda x: datetime.datetime.strptime(x,'%Y-%m-%dT%H:%M:%S-0300').date())
		self.news_df['rankings'] = self.news_df[['tag_array','date_object']].apply(lambda x: self.__calculate_content_based_rank(x,vector,date),axis=1)
		self.news_df = self.news_df.sort_values(by=['rankings'],ascending=False)
		print(self.news_df[['title','rankings']].head(5))

news_rec = NewsRecommender()
news_rec.train_content_based()
news_rec.recommend_content(5,'Jair Bolsonaro',datetime.date(2019,4,18))



