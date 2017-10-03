import numpy as np
from scipy import sparse
import operator
from sklearn.metrics import mean_squared_error
import datetime as dt
import random

class Recommender:
    
    def __init__(self, ratings, kind):
        self.ratings = ratings
        self.kind = kind
        self.nusers = ratings.shape[0]
        self.nitems = ratings.shape[1]
        self.similarity = None
        self.user_predictions = np.zeros((self.nusers, self.nitems))

    def get_similarity(self):
        """
        Get the user-to-user or item-to-item similarity matrix,
        using the ratings matrix and similarity type as input.

        :param self: instance of the Recommender class
        :returns: void
        """
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
        """
        Get the probability that a user will be interested
        in some item.

        :param self: instance of the Recommender class
        :returns: void
        """
        print('Getting user predictions...')

        if self.kind == 'user':
            self.user_predictions = self.similarity.dot(self.ratings)\
                    / np.array([np.abs(self.similarity).sum(axis=1)]).T

        elif self.kind == 'item':
            self.user_predictions = self.ratings.dot(self.similarity)\
                    / np.array([np.abs(self.similarity).sum(axis=1)])

        print('done!')

    def get_user_recs(self, user):
        """
        Get a ranked list of recommendations for a specific user.
        The ranking is done with high recommendation relevance corresponding to low indices.

        :param self: instance of the Recommender class
        :param user: the user number corresponding to a row of the ratings matrix
        :returns: ranked list of item numbers corresponding to columns of the ratings matrix
        """
        rec_dict = {}
        for i in range(self.nitems):
            rec_dict[i] = self.user_predictions[user,i]

        # sort the list in order of decreasing predicted ratings
        recs_sorted = sorted(rec_dict.items(), key=operator.itemgetter(1), reverse=True)

        # get recommendations
        # excluding items already selected
        user_recommendations = [x[0] for x in recs_sorted]
        user_items = np.nonzero(self.ratings[user])
        for i in user_items[0]:
            user_recommendations.remove(i)
        
        return user_recommendations

    def get_sim_items(self, item):
        """
        Get a ranked list of similar items using the collaborative filtering technique.
        This method requires the similarity type to be item-to-item.
        The ranking is done with high similarity corresponding to low indices.

        :param self: instance of the Recommender class
        :param item: item number corresponding to a column of the ratings matrix
        :returns: ranked list of item numbers corresponding to columns of the ratings matrix
        """
        if self.kind != 'item':
            print('Item similarity matrix not available!')
            return

        item_dict = {}
        for i in range(self.nitems):
            item_dict[i] = self.similarity[item,i]

        # sort in order of decreasing similarity
        items_sorted = sorted(item_dict.items(), key=operator.itemgetter(1), reverse=True)

        return [x[0] for x in items_sorted]



def train_test_split(ratings, nfolds, iteration):
    """
    Split the ratings matrix into disjoint training and test sets.
    The sets are disjoint with respect to the nonzero entries of the ratings matrix.
    Every nth nonzero entry is set to zero in the training matrix and set to a nonzero value in the test matrix.

    :param ratings: ratings matrix
    :param nfolds: the total number of cross validation folds
    :param iteration: the fold number used for this split
    :returns: disjoint matrices train and test with the same dimensions as the ratings matrix
    """
    test = np.zeros(ratings.shape)
    train = ratings.copy()

    holdout = np.transpose(np.nonzero(ratings))[iteration::nfolds]

    for i in range(len(holdout)):
        train[holdout[i][0], holdout[i][1]] = 0
        test[holdout[i][0], holdout[i][1]] = ratings[holdout[i][0], holdout[i][1]]

    # test and training are truly disjoint
    assert(np.all((train * test) == 0)) 
    return train, test

def get_roc(pred, actual, nrecitems):
    """
    Get the standard validation metrics, given a set of predictions and the ground truth.

    :param pred: Matrix containing item predictions for each user.  If corresponding to an N-item list of recommendations, this matrix will have N elements set to unity for each user, with zeros elsewhere.
    :param actual: Matrix containing the ground truth for each user.
    :param nrecitems: Number of recommendations in list.
    :returns: validation metrics.  True positives, false positives, true negatives, false negatives, precision, and recall.
    """

    # recommendation list with N items
    N = nrecitems

    for u in range(len(pred)):
        # indices of top N elements
        ind = sorted(range(len(pred[u])), key=lambda i: pred[u][i])[-N:]

        # set these indices to unity, else to zero
        # (this is redundant with the run_validation method)
        for i in range(len(pred[u])):
            pred[u][i] = 0
        for i in ind:
            pred[u][i] = 1

    tp = 0.0
    fp = 0.0
    tn = 0.0
    fn = 0.0
    precision = 0.0
    recall = 0.0
    print('calculating roc...')
    for i in range(pred.shape[0]):

        tpu = 0
        fnu = 0
        fpu = 0

        for j in range(pred.shape[1]):
            pred_value = pred[i,j]
            actual_value = actual[i,j]
            if( pred_value == 0 ):
                if( actual_value == 1 ):
                    fn += 1.0
                    fnu += 1.0
                else:
                    tn += 1.0
            elif( pred_value == 1):
                if( actual_value == 1 ):
                    tp += 1.0
                    tpu += 1.0
                else:
                    fp += 1.0
                    fpu += 1.0

        if tpu + fpu != 0:
            precision += tpu/(tpu+fpu)
        if tpu + fnu != 0:
            recall += tpu/(tpu+fnu)
    
    return tp, fp, tn, fn, precision/pred.shape[0], recall/pred.shape[0]


def run_validation(ratings, kind, nfolds, nrecitems=10, onefold=False):
    """
    Validate the recommender system by holding out wishlist and collection entries.

    :param ratings: the ratings matrix
    :param kind: type of recommendations (user-to-user, item-to-item, popular items, random items)
    :param nfolds: number of divisions of the wishlist/collection entries
    :param nrecitems: number of items recommended per user
    :param onefold: run a quicker validation with one holdout set instead of k-fold cross validation
    :returns: validation metrics (true positives, false positives, true negatives, false negatives, precision, recall)
    """
    print('Running validation with N =', nrecitems, '...')

    tp_sum = 0
    fp_sum = 0
    tn_sum = 0
    fn_sum = 0
    precision_sum = 0
    recall_sum = 0
    for fold in range(nfolds):
        print('... fold', fold+1, '/', nfolds)

        train, test = train_test_split(ratings, nfolds, fold)
        if( kind == 'popular' ):
            preds = np.zeros(train.shape)
            sums = np.sum(train, axis=0)

            # indices of top elements
            ind = sorted(range(len(sums)), key=lambda i: sums[i])[-nrecitems:]

            preds[:, ind] = 1

        elif( kind == 'random' ):
            max_item = train.shape[1]
            preds = np.zeros(train.shape)
            for u in range(len(preds)):
                preds[u][np.random.randint(low=0,high=max_item,size=nrecitems).tolist()] = 1

        else:
            rec = Recommender(train, kind)
            rec.get_similarity()
            rec.get_predictions()

            preds = rec.user_predictions # fix this?
            nans = np.isnan(preds)
            preds[nans] = 0

            # remove items already selected by user
            print('removing items already selected by user...')
            bindices_nonzero = (train != 0)
            preds[bindices_nonzero] = 0

        tp, fp, tn, fn, precision, recall = get_roc(preds, test, nrecitems)
        tp_sum += tp
        fp_sum += fp
        tn_sum += tn
        fn_sum += fn
        precision_sum += precision
        recall_sum += recall

        # run quicker validation
        # with one holdout set
        if onefold == True:
            break

    print('TP, FP, TN, FN = ', tp_sum, fp_sum, tn_sum, fn_sum)
    print('Precision:', tp_sum/(tp_sum+fp_sum) )
    print('True positive rate:', tp_sum/(tp_sum+fn_sum) )
    print('False positive rate:', fp_sum/(fp_sum+tn_sum) )
    print('User precision:', precision_sum )
    print('User recall:', recall_sum )

    return tp_sum, fp_sum, tn_sum, fn_sum, precision_sum, recall_sum
