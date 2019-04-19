import json
import pandas as pd

with open('data/news500.json','r') as f:
	news_json = json.load(f)
	news_json = news_json['data']['news']['entities']

df = pd.DataFrame(news_json)
print(df.columns)
