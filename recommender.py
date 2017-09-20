import numpy as np
from scipy import sparse
import operator
from sklearn.metrics import mean_squared_error

class Recommender:
    
    def __init__(self, ratings, kind):
        self.ratings = ratings
        self.kind = kind
        self.nusers = ratings.shape[0]
        self.nitems = ratings.shape[1]
        self.similarity = None
        self.user_predictions = np.zeros((self.nusers, self.nitems))

    def get_similarity(self):
        print('Calculating', self.kind, 'based similarity matrix...')

        # to avoid divide by zero
        eps = 1e-9

        # convert to sparse matrix
        sratings = sparse.csr_matrix(self.ratings)

        if self.kind == 'user':
            ssim = sratings.dot(sratings.T)

        elif (self.kind == 'item') or (self.kind == 'itemalt'):
            ssim = sratings.T.dot(sratings)

        sim = ssim.todense()
        norms = np.array([np.sqrt(ssim.diagonal()) + eps])

        self.similarity = np.array(sim / norms / norms.T)
        print('done!')


    def get_predictions(self):
        print('Getting user predictions...')

        if self.kind == 'user':
            self.user_predictions = self.similarity.dot(self.ratings)\
                    / np.array([np.abs(self.similarity).sum(axis=1)]).T

        elif self.kind == 'item':
            self.user_predictions = self.ratings.dot(self.similarity)\
                    / np.array([np.abs(self.similarity).sum(axis=1)])

        elif self.kind == 'itemalt':
            for u in range(self.nusers):
                if( u%1000 == 0 ):
                    print('... user', u)
                upred = np.zeros(self.nitems)
                for i in range(self.nitems):
                    if( self.ratings[u][i] != 0 ):
                        upred += self.ratings[u][i] * np.array(self.similarity[i,:])[0]
                self.user_predictions[u] = upred
        print('done!')


    def get_user_recs(self, user):

        rec_dict = {}
        for i in range(self.nitems):
            rec_dict[i] = self.user_predictions[user,i]

        # sort the list in order of decreasing predicted ratings
        recs_sorted = sorted(rec_dict.items(), key=operator.itemgetter(1), reverse=True)

        # get recommendations
        # excluding items already covered
        user_recommendations = [x[0] for x in recs_sorted]
        user_items = np.nonzero(self.ratings[user])
        for i in user_items[0]:
            user_recommendations.remove(i)
        
        return user_recommendations

def train_test_split(ratings, nfolds, iteration):

    test = np.zeros(ratings.shape)
    train = ratings.copy()

    holdout = np.transpose(np.nonzero(ratings))[iteration::nfolds]

    for i in range(len(holdout)):
        train[holdout[i][0], holdout[i][1]] = 0
        test[holdout[i][0], holdout[i][1]] = ratings[holdout[i][0], holdout[i][1]]

    # test and training are truly disjoint
    assert(np.all((train * test) == 0)) 
    return train, test

def get_mse(pred, actual):
    # ignore nonzero terms.
    pred = pred[actual.nonzero()].flatten()
    actual = actual[actual.nonzero()].flatten()
    return mean_squared_error(pred, actual)

def run_validation(ratings, kind, nfolds):
    print('Running validation...')

    mse_sum = 0
    for fold in range(nfolds):
        print('... fold', fold+1, '/', nfolds)

        train, test = train_test_split(ratings, nfolds, fold)
        rec = Recommender(train, kind)
        rec.get_similarity()
        rec.get_predictions()

        preds = rec.user_predictions # fix this?
        nans = np.isnan(preds)
        preds[nans] = 0

        mse_sum += get_mse(preds, test)

    mse_sum /= nfolds
    print('MSE:', mse_sum)
    return mse_sum
