
import json
import queue
from redis import StrictRedis
from surprise import dump
from surprise import Dataset
from surprise import Reader
from surprise.model_selection import train_test_split
import pandas as pd
import numpy as np

from celery import Celery
from anime import get_top_n
import logging

"""
The structure of the dorectory. Note: please set the worker.py and anime dataset directory in the same path. 
.
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
# logger = logging.getLogger('celery')
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter("TIME=%(asctime)s,[%(levelname)s]:%(message)s")
ch = logging.StreamHandler()
ch.setFormatter(formatter)
logger.addHandler(ch)


# Create a Celery app, providing a name and the URI to the message broker
app = Celery('tasks', backend='redis://localhost/0', broker='redis://localhost/1')


def get_cos_similar_multi(v1: list, v2: list):
    # Calculate the cosine similarity between a vector and all candidate vectors
    # Ref: https://www.jianshu.com/p/613ff3b1b4a8
    num = np.dot([v1], np.array(v2).T)  # 向量点乘
    denom = np.linalg.norm(v1) * np.linalg.norm(v2, axis=1)  # 求模长的乘积
    res = num / denom
    res[np.isneginf(res)] = 0
    return 0.5 + 0.5 * res


# Create a task using the app.task decorator
@app.task
def make_recommendation():
    _, model = dump.load('anime_model.pickle')
    reader = Reader(rating_scale=(1, 10))

    # the model loaded from file doesn't contain the relationship between raw id and inner id, so to handle new users we have to load trainset again
    user = pd.read_csv('./anime/anime_ratings.dat', encoding='utf-8', delimiter="\t")
    trainset = user[user['User_ID'].isin(user['User_ID'].unique()[:4000])]
    traindata = Dataset.load_from_df(
        trainset[['User_ID', 'Anime_ID', 'Feedback']], reader
    ).build_full_trainset()
    logger.info("Loaded model.")

    # Creating a Publisher
    # Get a connection to Redis
    queue = StrictRedis(host='localhost', port=6379)
    logger.info("Got a connection to Redis.")

    # Creating a Subscriber
    # Connect and subscribe
    pubsub = StrictRedis(host='localhost', port=6379).pubsub()
    pubsub.subscribe('anime.request')
    logger.info("Subscribed anime.request.")
    # The first message you receive will be a confirmation of subscription
    message = pubsub.get_message()
    # {'pattern': None, 'type': 'subscribe', 'channel':'testing', 'data':1L}

    # The subsequent messages are those from the publisher(s)
    for message in pubsub.listen():
        # print(message)
        logger.info("New message come")
        new_ratings = eval(message['data'].decode("utf-8"))
        new_df = pd.DataFrame(new_ratings)

        iids = [traindata.to_raw_iid(i) for i in traindata.all_items()]

        if new_df.shape[0] == 0:
            # the user input nothing (no information on the initial ratings), recommend the most popular animes to the user
            new_data = pd.DataFrame(
                {
                    'User_ID': [5001] * len(iids),
                    'Anime_ID': iids,
                    'Feedback': [0] * len(iids),
                }
            )

        else:
            # the user input some ratings, use these ratings to find the most similar user to make recommendations
            # generate all items for this uesr to predict

            df = pd.DataFrame(
                {
                    'User_ID': [new_df.iloc[0].at['User_ID']] * len(iids),
                    'Anime_ID': iids,
                }
            )

            new_data = pd.merge(
                df, new_df, on=['User_ID', 'Anime_ID'], how='outer'
            ).fillna(0)

            # project the new user into the user feature matrix using the rating information of the new user
            # Ref: https://dorianzi.github.io/2019/03/20/recommender-system-by-SVD/
            new_user_rating = np.array(new_data['Feedback'])
            new_user_feature = new_user_rating.dot(model.qi)

            # find the most similar user with the new user using cosine similarity
            simialr_user = traindata.to_raw_uid(
                np.argmax(get_cos_similar_multi(new_user_feature, model.pu))
            )
            logger.info("The most similar user is {}".format(simialr_user))

            # use similar user to make predictions
            new_data['User_ID'] = simialr_user

            # not make preds for animes that new user has rated
            new_data = new_data.drop(
                new_data[new_data['Anime_ID'].isin(new_df['Anime_ID'].unique())].index
            )

        new_data = Dataset.load_from_df(
            new_data[['User_ID', 'Anime_ID', 'Feedback']], reader
        )

        _, new_data = train_test_split(new_data, test_size=1.0)

        pred = model.test(new_data)
        pred = get_top_n(pred, n=10)
        logger.info("Made preds successfully.")

        pred_str = json.dumps(pred)
        queue.publish("anime.response", pred_str)  # .encode('utf-8)
        logger.info("Published the prediction result to Redis.")
