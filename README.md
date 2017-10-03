# Recommender System

This project leverages collection and wishlist data to generate personalized recommendations for Covetly users.  The process of generating recommendations is summarized in three jupyter notebooks:
1.  [Read Database](1-ReadDatabase.ipynb):  Collection and wishlist data is aggregated from the MongoDB database and saved into local csv files.
2.  [Recommender Demo](2-RecommenderDemo.ipynb):  The data is inserted into a ratings matrix, which is used to train the recommender system.  The recommender employs a collaborative filtering algorithm with a cosine similarity metric.  These core components of the recommender system are encapsulated in the Recommender class in `recommender.py`.
3.  [Validation](3-Validation.ipynb):  The recommender system is validated by computing standard metrics such as precision and recall.  The performance is compared to baseline approaches, such as recommending popular items in the Covetly store, or recommending a random list of items.

The notebooks can be run directly -- they should require only several standard python packages (NumPy, SciPy, Pandas, Matplotlib).  Access to the MongoDB database requires a file `db.txt` with the appropriate paths.

For a demonstration of this recommender system, see the web application [here](http://collectorizer.nathanmirman.com).
