# Collectorizer

Collectorizer is a recommender system for collectors.  It was developed as a consulting project for [Covetly](https://www.covetlyapp.com), a startup mobile-app marketplace for buying and selling collectibles.

Users of the Covetly app specify a collection of items they already own, and a wishlist of items that they may be interested in.  Collectorizer utilizes this information to generate personalized recommendations for each user.

The process of generating recommendations is summarized in three jupyter notebooks:
1.  [Read Database](1-ReadDatabase.ipynb):  Collection and wishlist data is aggregated from the Covetly MongoDB database and saved into local csv files.
2.  [Recommender Demo](2-RecommenderDemo.ipynb):  The data is inserted into a ratings matrix, which is used to train the recommender system.  The recommender employs a collaborative filtering algorithm with a cosine similarity metric.  These core components of the recommender system are encapsulated in the Recommender class in `recommender.py`.
3.  [Validation](3-Validation.ipynb):  The recommender system is validated by computing standard metrics such as precision and recall.  The performance is compared to baseline approaches, such as recommending popular items in the Covetly store, or recommending a random list of items.

A further description and demonstration of Collectorizer is available at [collectorizer.nathanmirman.com](http://collectorizer.nathanmirman.com).
