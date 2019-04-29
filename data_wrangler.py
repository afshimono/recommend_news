import json
import pandas as pd
import random
from pandas.io.json import json_normalize
# https://www.kaggle.com/sohier/introduction-to-the-bq-helper-package
import os
import glob

pd.set_option('display.max_colwidth', -1)		#this is used to show all columns on prints

raw_data_folder = 'raw_data/news'

#See all News files in one folder and generate a CSV file with consolidated data
def create_news_csv(news_folder):
	for filename in glob.glob(os.path.join(news_folder, '*.json')):
		with open(filename,'r') as f:
			news_json = json.load(f)
			news_json = news_json['data']['news']['entities']
			df = pd.DataFrame(news_json)
			df = df.drop(['url','__typename','topper', 'fieldCategories', 'fieldChannels','fieldMainImage','tags'],axis=1)
			tags_df = json_normalize(data=news_json, record_path=['tags'],meta='entityId',errors='ignore')		
			tags_df = tags_df.rename(index=str, columns={"entityId": "newsEntityId"})
			tags_df = tags_df.to_json(orient='records')
			tags_df = json_normalize(data=json.loads(tags_df),errors='ignore')
			tags_df = tags_df[['entity.name','newsEntityId']].groupby('newsEntityId')['entity.name'].apply(list)
			df = df.merge(tags_df,left_on='entityId',right_on='newsEntityId').rename(index=str, columns={"entity.name": "tags"})
			df['tags'] = df['tags'].apply(', '.join)
			if(os.path.isfile('wrangled_data/news_df.csv')):
				#print('Appending')
				df.to_csv('wrangled_data/news_df.csv',index=False,encoding='utf-8',mode='a', header=False)
			else:
				#print('Creating')
				df.to_csv('wrangled_data/news_df.csv',index=False,encoding='utf-8',mode='w', header=True)

#This function receives the number x of pages visited by a user and returns an array with x random page ids
def generate_random_page_ids(x,options):
	result = []
	while len(result)<x:
		item = options[random.randint(0,len(options)-1)]
		if item not in result:
			result.append(item)
	return result


#This function will read an analytics json and create a CSV with relevant filtered information.
#It also will change each landing page from the analytics default to the newsEntityId
def create_analytics_csv():
	json_list = []
	with open('raw_data/analytics/analytics.json','r',encoding="utf-8") as f:
		for line in f:
			json_list.append(json.loads(line))
		analytics_df = pd.DataFrame(json_list)

		totals_json = json.loads(analytics_df[['totals','visitId']].to_json(orient='records'))
		totals_df = json_normalize(data=totals_json,errors='ignore')
		#print(totals_df.columns)
		traffic_json = json.loads(analytics_df[['trafficSource','visitId']].to_json(orient='records'))
		traffic_df =  json_normalize(data=traffic_json,errors='ignore')
		traffic_df = traffic_df[['visitId','trafficSource.keyword','trafficSource.source',]]
		#print(traffic_df.columns)
		device_json = json.loads(analytics_df[['device','visitId']].to_json(orient='records'))
		device_df = json_normalize(data=device_json,errors='ignore')
		device_df = device_df[['visitId','device.browser','device.isMobile','device.language','device.operatingSystem']]
		#print(device_df.columns)
		geo_json = json.loads(analytics_df[['geoNetwork','visitId']].to_json(orient='records'))
		geo_df = json_normalize(data=geo_json,errors='ignore')
		geo_df = geo_df[['visitId','geoNetwork.city','geoNetwork.country','geoNetwork.networkDomain']]
		#print(geo_df.columns)

		analytics_df = analytics_df[['visitId','visitStartTime','date','visitNumber']]
		analytics_df = analytics_df.merge(totals_df,left_on='visitId', right_on='visitId')
		analytics_df = analytics_df.merge(traffic_df,left_on='visitId', right_on='visitId')
		analytics_df = analytics_df.merge(device_df,left_on='visitId', right_on='visitId')
		analytics_df = analytics_df.merge(geo_df,left_on='visitId', right_on='visitId')
		#print(analytics_df.columns)

		news_df = pd.read_csv('wrangled_data/news_df.csv')
		option_list = news_df['entityId'].tolist()
		analytics_df['pages_visited'] = analytics_df['totals.pageviews'].apply(lambda x: generate_random_page_ids(int(x),options=option_list))
		analytics_df.to_csv('wrangled_data/analytics_df.csv',index=False,encoding='utf-8',mode='w', header=True)


#create_news_csv(raw_data_folder)
create_analytics_csv()
