import pandas as pd
from surprise import SVD
from surprise import Dataset
from surprise import Reader
from surprise.model_selection import cross_validate
from surprise.model_selection import train_test_split
from collections import defaultdict

from sklearn.metrics import accuracy_score
from surprise import dump

import pprint

import logging

"""
The structure of the dorectory. Note: please set the anime.py and anime dataset directory in the same path. 
.
├── IEMS5780_A4_1155162614_KunWANG.pdf
├── worker.py
├── anime
│   ├── README
│   ├── anime_history.dat
│   ├── anime_info.dat
│   └── anime_ratings.dat
├── anime.py
├── anime.zip
├── anime_model.pickle
├── bot.py

"""

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter("TIME=%(asctime)s,[%(levelname)s]:%(message)s")
ch = logging.StreamHandler()
ch.setFormatter(formatter)
logger.addHandler(ch)


def preprocess_data(user):
    """
    Split user rating data into trainset and testset for sebsequent process with Surprise library.

    :param user: pandas Dataframe, all data including trainset and testset

    :returns (trainset, testset): tuple

    """

    # Use the first 4000 users as training data, and leave the remaining (less than 1000) users as a testing set.
    trainset = user[user['User_ID'].isin(user['User_ID'].unique()[:4000])]
    testset = user[user['User_ID'].isin(user['User_ID'].unique()[4000:])]

    # uids = user['User_ID'].unique()[4000:]
    # iids = user['Anime_ID'].unique()
    # len_iids = len(iids)
    # df = pd.DataFrame(columns=['User_ID', 'Anime_ID'], dtype = np.int64)
    # for i in range(len(uids)):
    #     df_tmp = pd.DataFrame({'User_ID': [uids[i]]*len_iids, 'Anime_ID':iids})
    #     df = pd.concat([df, df_tmp], ignore_index=True)

    # testset =  pd.merge(df, testset, on=['User_ID', 'Anime_ID'], how='outer')

    # A reader is still needed but only the rating_scale param is requiered.
    reader = Reader(rating_scale=(1, 10))

    # The columns must correspond to user id, item id and ratings (in that order).
    traindata = Dataset.load_from_df(
        trainset[['User_ID', 'Anime_ID', 'Feedback']], reader
    )
    testdata = Dataset.load_from_df(
        testset[['User_ID', 'Anime_ID', 'Feedback']], reader
    )
    _, testdata = train_test_split(testdata, test_size=1.0)

    return (traindata, testdata)


def get_top_n(predictions, n=10):
    """Return the top-N recommendation for each user from a set of predictions.

    Args:
        predictions(list of Prediction objects): The list of predictions, as
            returned by the test method of an algorithm.
        n(int): The number of recommendation to output for each user. Default
            is 10.

    Returns:
    A dict where keys are user (raw) ids and values are lists of tuples:
        [(raw item id, rating estimation, true rating), ...] of size n.

    Ref: https://surprise.readthedocs.io/en/stable/FAQ.html#how-to-get-the-top-n-recommendations-for-each-user
    """

    # First map the predictions to each user.
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est, true_r))

    # Then sort the predictions for each user and retrieve the n highest ones.
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]
    return top_n


def compute_matrix(top_n):
    """
    Compute (i) accuracy based on the testing set using top-10 predictions.

    :param top_n: A dict where keys are user (raw) ids and values are lists of tuples: [(raw item id, rating estimation, true rating), ...] of size n.

    :returns accuracy: float
    """

    # treate ratings larger than 5 as positive (1), otherwise, negative (0)
    actual = []
    predicted = []
    for uid, user_ratings in top_n.items():
        for top_item in user_ratings:
            if top_item[1] > 5:
                predicted.append(1)
            else:
                predicted.append(0)

            if top_item[2] is not None and top_item[2] > 5:
                actual.append(1)
            else:
                actual.append(0)
    # from sklearn.metrics import classification_report
    # target_names = ['wrong pred', 'correct pred']
    # matrix = classification_report(
    #     actual, predicted, target_names=target_names, output_dict=True
    # )
    # pprint.pprint(matrix)
    # return matrix['accuracy'], matrix['correct pred']['f1-score']
    return accuracy_score(actual, predicted)


def search_parm(traindata):
    """
    Search best parameters for SVD.
    """
    from surprise.model_selection import GridSearchCV

    param_grid = {
        "n_epochs": [5, 10, 15, 20, 30, 40, 50, 100],
        "lr_all": [0.001, 0.002, 0.005],
        "reg_all": [0.02, 0.08, 0.4, 0.6],
    }

    gs = GridSearchCV(SVD, param_grid, measures=["rmse"], refit=True, cv=5)
    gs.fit(traindata)
    training_parameters = gs.best_params["rmse"]
    return training_parameters


if __name__ == "__main__":
    # load and split data
    logger.info("Loading and splitting data...")

    user = pd.read_csv('./anime/anime_ratings.dat', encoding='utf-8', delimiter="\t")

    (traindata, testdata) = preprocess_data(user)
    logger.info("OK")

    # Run 5-fold cross-validation and print results.
    logger.info("Run 5-fold cross-validation...")
    cross_validate(SVD(), traindata, measures=['RMSE'], cv=5, verbose=True)

    # search best parameters for SVD model training
    # training_parameters = search_parm(traindata)
    training_parameters = {
        'n_epochs': 100,
        'lr_all': 0.002,
        'reg_all': 0.08,
    }  # To save time

    # Use SVD algorithm.
    algo = SVD(
        n_epochs=training_parameters['n_epochs'],
        lr_all=training_parameters['lr_all'],
        reg_all=training_parameters['reg_all'],
    )
    # algo = SVD()

    # train model using SVD
    logger.info("Training...")
    algo.fit(traindata.build_full_trainset())
    logger.info("OK")

    # make predictions on testset
    pred = algo.test(testdata)

    # get top-10 predictions
    top_10 = get_top_n(pred, n=10)

    # compute accuracy based on the testing set using top-10 predictions.
    logger.info("Compute accuracy based on the testing set using top-10 predictions.")
    # (accuracy, f_measure) = compute_matrix(top_10)
    accuracy = compute_matrix(top_10)
    print("accuracy:{}".format(accuracy))

    # save the model as anime_model.pickle
    logger.info("Saving model...")
    dump.dump('anime_model.pickle', predictions=pred, algo=algo)
    logger.info("Done")
