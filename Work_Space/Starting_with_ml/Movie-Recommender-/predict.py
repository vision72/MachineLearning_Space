import numpy as np
from lightfm.datasets import fetch_movielens
from lightfm import LightFM

data = fetch_movielens(min_rating=4.0)

print repr(data['train'])
print repr(data['test'])

model = LightFM(loss='warp')

model.fit(data['train'], epochs=30, num_threads=2)

def sample_recommendation(model, data, user_ids):
	n_users, n_movies = data['train'].shape
	for user_id in user_ids:
		known_positive = data['item_labels'][data['train'].tocsr()[user_id].indices]

		score = model.predict(user_id, np.arange(n_movies))
		top_items = data['item_labels'][np.argsort(-score)]

		print "User %s" % user_id
		print "			known_positives:"

		for x in known_positive[:3]:
			print "			%s" %x

		print "			recommendation:"

		for y in top_items[:3]:
			print "			%s" %y
sample_recommendation(model, data, [3, 25, 450])
