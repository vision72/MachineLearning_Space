import tweepy
import pandas as pd
from textblob import TextBlob

consumer_key = 'TagLmb4lBWzP3oZKFFodg7ERv'
consumer_secret = 'pFsdv9yiVlNBFMSeKyQ89I35LjQOldmFTHWl8mDUrRVQ99z6gA'

access_token = '911855257168343040-vP4nJ5MPs90lQUKdn1jRFh0QR2PVYRl'
access_token_secret = 'g3sroIKJupcZry4mI7b4uJowuuVPxqKNup9zL482kfF3h'

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth)

tweets = api.search('deep learning')

data = []

for x in tweets:
	# print x.text.encode("utf-8")
	analysis = TextBlob(x.text)
	# print analysis.sentiment
	data.append(analysis.sentiment)
# data = '\n'.join(data)
print data

my_submission = pd.DataFrame({'Sentiment: ': [x for x in data] })
my_submission.to_csv('submission.csv', index = False)
