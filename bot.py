from telegram.ext.updater import Updater
from telegram.update import Update
from telegram.ext.callbackcontext import CallbackContext
from telegram.ext.commandhandler import CommandHandler
from telegram.ext.messagehandler import MessageHandler
from telegram.ext.filters import Filters
import telegram

import pandas as pd
import numpy as np
import json
from redis import StrictRedis
import ast

from worker import make_recommendation


import logging

"""
The structure of the dorectory. Note: please set the bot.py and anime dataset directory in the same path. 
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
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter("TIME=%(asctime)s,[%(levelname)s]:%(message)s")
ch = logging.StreamHandler()
ch.setFormatter(formatter)
logger.addHandler(ch)

metadata = pd.read_csv('./anime/anime_info.dat', encoding='utf-8', delimiter="\t")


# Creating a Publisher
# Get a connection to Redis
queue = StrictRedis(host='localhost', port=6379)

# Creating a Subscriber
# Connect and subscribe
pubsub = StrictRedis(host='localhost', port=6379).pubsub()
pubsub.subscribe('anime.response')


global bot
bot = telegram.Bot(token="your_token") # !!! Replace ``your_token`` field with your own bot's token


class State:
    def __init__(self):
        self.state = "initial"
        self.uid = 5001


def get_anime_info(anime_id):

    """
    Given the anime id, return name of this anime.
    """
    # anime_info = metadata[metadata['anime_ids'] == anime_id][
    #     ['name', 'genre', 'type', 'episodes', 'rating', 'members']
    # ]
    # return anime_info.to_dict(orient='records')
    anime_name = metadata[metadata['anime_ids'] == anime_id].iloc[0].at['name']
    return anime_name


def rand_list(length):
    """
    Randomly generate a list of animes of length length.

    :param length: int, the length of the list

    :returns animes_id_name: a list where the items are dict where keys are Anime_ID (raw) ids and values strings which are full name of the animes.
        Example: [{'anime_ids': 1, 'name': 'D.N.Angel'}, {'anime_ids': 3, 'name': 'Bokusatsu Tenshi Dokuro-chan'}]
    """
    anime_ids = metadata['anime_ids'].unique()
    iid_list = np.random.choice(anime_ids, size=length, replace=False, p=None)
    animes_id_name = metadata[metadata['anime_ids'].isin(iid_list)][
        ['anime_ids', 'name']
    ]
    return animes_id_name.to_dict(orient='records')


def start(update: Update, context: CallbackContext):
    update.message.reply_text(
        "Hello sir, welcome to IEMS5780 Assignment 4 demo bot. Please write\
        /help to see the commands available."
    )


def help(update: Update, context: CallbackContext):
    update.message.reply_text(
        """Available Commands :-
    /recommend_animes - To recommend the animes based on user input (multiple ratings for a list of animes).
    Otherwise - cannot interpret.
    """
    )


def recommend_animes(update: Update, context: CallbackContext):
    if s.state == 'initial':
        update.message.reply_text(
            "How many animes would you like to give a rating? Please input a number. We will randomly generate a list of animes for you to give ratings! \n (Input 0 if you don't want to give any ratings.)"
        )
        s.state = "waiting input of length"


def pred():
    logger.info("waiting input of predictions")
    # The first message you receive will be a confirmation of subscription
    message = pubsub.get_message()
    # {'pattern': None, 'type': 'subscribe', 'channel':'testing', 'data':1L}
    while True:
        message = pubsub.get_message()
        if message:
            # pred = ast.literal_eval(message['data'].decode("utf-8"))
            top_10 = eval(message['data'])
            print(top_10)
            break
        else:
            continue
    text = 'The top-10 recommended amines are listed as follows.\n'
    for i in range(10):
        text += (
            str(i)
            + ': '
            + str(get_anime_info(top_10[list(top_10.keys())[0]][i][0]))
            + '\n'
        )
    bot.sendMessage(chat_id=s.id, text=text)
    s.state = 'initial'


def general(update: Update, context: CallbackContext):
    if s.state == "waiting input of length":
        logger.info("waiting input of length")
        length = update.message.text
        try:
            s.length = int(length)
        except:
            update.message.reply_text("Please input a number.")

        s.animes_id_name = rand_list(s.length)

        text = 'Could you please give ratings for the following animes? \n'
        for i in range(s.length):
            text += str(i) + ': ' + s.animes_id_name[i]['name'] + '\n'
        text += (
            'You should input a list of ratings of length '
            + str(s.length)
            + ', e.g. [1,5,...,8] \n'
            + '(Input [] if you do not want to give any ratings.)'
        )
        update.message.reply_text(text)

        s.state = "waiting input of ratings"

        logger.info("ok")

    elif s.state == "waiting input of ratings":
        logger.info("waiting input of ratings")
        rating_list = update.message.text
        s.id = update.message.chat_id
        try:
            rating_list = eval(rating_list)

        except:
            update.message.reply_text(
                "Sorry, please input a list like message of specific length, e.g. [1,5,8] of length 3"
            )
        s.ratings = []
        for i in range(s.length):
            dict_tmp = {}
            dict_tmp['User_ID'] = s.uid
            dict_tmp['Anime_ID'] = s.animes_id_name[i]['anime_ids']
            dict_tmp['Feedback'] = rating_list[i]
            s.ratings.append(dict_tmp)
        data_str = json.dumps(s.ratings)
        # example of data_str: '[{"User_ID": 5001, "Anime_ID": 7327, "Feedback": 4}, {"User_ID": 5001, "Anime_ID": 3727, "Feedback": 8}]'

        # Publish a message to a channel called anime.request
        queue.publish("anime.request", data_str.encode("utf-8"))

        logger.info("published ratings to anime.request.")
        pred()

    else:
        update.message.reply_text(
            "Sorry I can't recognize you , you said '%s'" % update.message.text
        )

    return


if __name__ == "__main__":

    # Provide your bot's token
    updater = Updater(
        "your_token", use_context=True
    ) # # !!! Replace ``your_token`` field with your own bot's token
    make_recommendation.delay()
    s = State()

    updater.dispatcher.add_handler(CommandHandler('start', start))
    updater.dispatcher.add_handler(CommandHandler('help', help))
    updater.dispatcher.add_handler(CommandHandler('recommend_animes', recommend_animes))

    updater.dispatcher.add_handler(MessageHandler(Filters.text, general))
    updater.start_polling()
