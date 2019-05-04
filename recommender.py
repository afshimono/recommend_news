#We will use Scikit-Surprise for Recommender Systems
#We will implement one Content Based and one Colaborative Filter recommendation systems,
#and use both to create a hybrid model.

import pandas as pd
import numpy as np
from sklearn.metrics import pairwise
import datetime
from surprise import SVD
from surprise import Dataset
from surprise.model_selection import cross_validate
from surprise import SVDpp
from surprise import accuracy
from surprise.model_selection import train_test_split
from surprise.model_selection import GridSearchCV
from surprise import Reader
from surprise import dump
import glob
import os
import datetime

class SimpleNewsRecommender:
	#Loads the serialized trained algorithm from disk
	def __load_algo(self):
		algo_location = glob.glob('state/algo_*')[0]
		_, loaded_algo = dump.load(algo_location)
		return loaded_algo


	def __init__(self,trained_model=False):
		if trained_model:
			self.news_df = pd.read_csv('wrangled_data/news_df.csv')
			self.analytics_df = pd.read_csv('wrangled_data/analytics_df.csv')
			self.position_dic = {}
			self.visit_user_ranking = pd.read_csv('state/visit_user_ranking.csv')
			self.algo = self.__load_algo()

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
	def recommend_content_based(self,n,tags,date):
		vector = self.__create_vector(tags,self.position_dic)
		self.news_df['date_object'] = self.news_df['publishedAt'].apply(lambda x: datetime.datetime.strptime(x,'%Y-%m-%dT%H:%M:%S-0300').date())
		self.news_df['rankings'] = self.news_df[['tag_array','date_object']].apply(lambda x: self.__calculate_content_based_rank(x,vector,date),axis=1)
		self.news_df = self.news_df.sort_values(by=['rankings'],ascending=False)
		return self.news_df.head(n)


	#Calculates the ranking of each page for each user.
	def __generate_ranking(self,x):
		#Rule applied: no time equals 0, 
		#less than 2 with time equals 1, 
		#between 2 and 4 with time equals 2, 
		#between 4 and 6 with time equals 3, 
		#larger than 6 with time equals 4
		if not x[1]:
			return 0
		elif x[0] < 2:
			return 1
		elif 2 <= x[0] < 4:
			return 2
		elif 4 <= x[0] < 6:
			return 3
		else: 
			return 4

	def train_collaborative_filtering(self, grid_search=False, gs_params = None):
		#transform page list in single value
		analytics_df_SVD = self.analytics_df.copy()
		analytics_df_SVD['ranking'] = analytics_df_SVD[['totals.pageviews','totals.timeOnSite']].apply(lambda x:self.__generate_ranking(x),axis=1)
		analytics_df_SVD = analytics_df_SVD['pages_visited'].apply(lambda x: pd.Series(eval(x)))\
			.stack()\
			.reset_index(level=1,drop=True)\
			.to_frame('pageId')\
			.join(analytics_df_SVD[['visitId','ranking']], how='left')
		analytics_df_SVD = analytics_df_SVD.dropna()
		analytics_df_SVD = analytics_df_SVD[['visitId','ranking','pageId']]
		analytics_df_SVD['pageId'] = analytics_df_SVD['pageId'].apply(lambda x:int(x))

		# Saves Matrix for later use
		analytics_df_SVD.to_csv('state/visit_user_ranking.csv')

		# A reader is still needed but only the rating_scale param is requiered.
		reader = Reader(rating_scale=(1, 4))

		# The columns must correspond to user id, item id and ratings (in that order).
		data = Dataset.load_from_df(analytics_df_SVD[['visitId', 'pageId', 'ranking']], reader)



		trainset, testset = train_test_split(data, test_size=.1)

		# If user desires to use GridSearch to find best params and algo
		if grid_search:
			if(not gs_params):
				param_grid = {'n_factors': [110, 120, 140, 160], 'n_epochs': [90, 100, 110], 'lr_all': [0.001, 0.003, 0.005, 0.008],\
		              'reg_all': [0.08, 0.1, 0.15]}
			else:
				param_grid=gs_params		    	
			gs = GridSearchCV(SVD, param_grid, measures=['rmse', 'mae'], cv=3)			
			gs.fit(data)
			algo = gs.best_estimator['rmse']
			print(gs.best_score['rmse'])
			print(gs.best_params['rmse'])

		## Comment next lines if you are searching the best params
		# We can now use this dataset as we please, e.g. calling cross_validate
		else:
			algo = SVD(n_factors=110, n_epochs=110, lr_all=0.008, reg_all=0.15)


		cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)


		algo.fit(trainset)
		test_pred = algo.test(testset)
		print("SVD : Test Set")
		accuracy.rmse(test_pred, verbose=True)

		# Dump algorithm 
		print('Saving trained algo...',end =" ")
		algo_list = glob.glob('state/algo_*')
		file_name = 'state/algo_'+ datetime.datetime.now().strftime("%Y_%B_%d__%Ih%M%p")
		dump.dump(file_name, algo=algo)
		for file in algo_list:
			os.remove(file)
		print('Done.')


	# This function predicts the best n suggestions to a given user based on the trained 
	# collaborative filtering recommender
	def recommend_collaborative_filtering(self, uid, n):
		#list of news
		news_list = self.news_df['entityId'].tolist()
		#finds which news were already seen by the viewer
		viewed_list = self.visit_user_ranking[self.visit_user_ranking['visitId']==int(uid)]['pageId'].tolist()
		for page in viewed_list:
			news_list.remove(page)
		result_list = []
		for page in news_list:
			result_list.append(self.algo.predict(int(uid),page))
		result_df = pd.DataFrame(result_list)
		#print(result_df.head(5))
		print(result_df.sort_values(by=['est'], ascending=False).join(self.news_df.set_index('entityId'),on='iid')[['title','est']].head(n))


news_rec = SimpleNewsRecommender(trained_model=True)
#news_rec.train_content_based()
#news_rec.recommend_content_based(5,'Jair Bolsonaro',datetime.date(2019,4,18))
#news_rec.train_collaborative_filtering()
news_rec.recommend_collaborative_filtering('1495138189',5)



