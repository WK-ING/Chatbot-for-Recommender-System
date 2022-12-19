# Chatbot-for-Recommender-System
 This is a chatbot based on Telegram for recommender system of  anime.  

## Table of Contents
- [Installing](#installing)
- [Background](#background)
    - [Training](#training-a-recommender-system-using-surprise)
    - [Deploy the model as worker](#deploying-the-model-as-some-worker-programs)
    - [Communicate with worker by Telegram](#communicating-with-your-worker-programs-in-the-telegram-bot-using-redis-message-queue)
- [Usage](#usage)
- [Framework](#system-architecture)
- [Demo](#demo)

## Installing

```shell
sudo apt update
sudo apt install python3 python3-pip
sudo apt-get install redis-server
pip install redis
sudo apt install python3-testresources
pip install celery
```

## Background
In this project, we are going to 
1. set up the environment for running the Celery workers with a Redis backend, 
2. train a Surprise recommender system for the anime dataset, 
3. write a worker program to run trained model and 
4. write a telegram bot to communicate the worker via the Redis message queue.

Data source and credit ([CaseRec@github](https://github.com/caserec/Datasets-for-Recommender-Systems/tree/master/Processed%20Datasets/Anime))

### Training a recommender system using Surprise
- Save program as ``anime.py``.
- Download the dataset to ``anime``.
- Use the first 4000 users as 5-fold cross validation, and leave the last 1000 users as a testing set.
- Compute the RMSE (root mean square error) for the 5-fold cross validation.
- Generate a SVD model using all training data, and save the model as ``anime_model.pickle`` using ``dump``.
- Compute (i) accuracy and (ii) F-measure based on the testing set using top-10 predictions.
    - For each of the top 10 predictions, if the actual score is greater than 5, it is regarded as a correct prediction.

### Deploying the Model as some worker programs
- Save the worker program as ``worker.py``.
- Load trained model.
- Use celery with a Redis backend, and subscribe to the channel ``“anime.request”``.
- After processing the requests from the channel ``“anime.request”``, publish the results to ``“anime.reponse”``.

### Communicating with your worker programs in the Telegram bot using Redis message queue
- Create a bot based on Telegram. [See](https://core.telegram.org/bots/features#botfather).
- Save the telegram bot as ``bot.py``.
- Create an interface that users can input multiple ratings for a list of animes.
    - Assume that the ratings will range between 1 and 10.
    - Assume that the names for the anime are correct and within the training data.
- Publish the ratings in the previous step to the ``Redis`` channel ``“anime.request”``.
- Subscribe to the ``“anime.response”``, and recommend the top 10 anime to the user after receiving the response.

## Usage
Open a command terminal.

1. start the redis server.
```shell
redis-server
```

2. start three concurrent workers.
```shell
celery -A worker worker --concurrency=3 --loglevel=info
```

3. Replace the ``your_token`` field in ``line 58`` and ``line 212`` of ``bot.py`` with your own bot's token. Start telegram services.
```shell
python3 bot.py
```
4. Open the Telegram app, find your own bot, send messages to it. 


## System Architecture
- ``anime.py`` file stores programs for data preparations and training the model.
- ``bot.py`` program stores telegram services.
- ``worker.py`` program stores backend worker programs.
- ``anime_model.pickle`` stores trained ``Surprise`` model.

![image](/image/framework.png)

## Demo
![image](/image/help-demo.png)
![image](/image/recommand-demo.png)