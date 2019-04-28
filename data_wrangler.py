import json
import pandas as pd
from pandas.io.json import json_normalize
# https://www.kaggle.com/sohier/introduction-to-the-bq-helper-package
import os
import glob

pd.set_option('display.max_colwidth', -1)		#this is used to show all data on prints

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
create_news_csv(raw_data_folder)
