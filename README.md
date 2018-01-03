# tweet-classifier
Machine learning algorithm to classify tweets as Hillary Clinton or Donald Trump. I used three methods:
1. Bag-of-words model (average accuracy: 86.4%, training time per cycle: 1 min 50 sec)
2. Embedded word vectors (average accuracy: 77.8%, training time per cycle: 1 min 35 sec)
3. Recurrent neutral network (in progress).

Dependencies:
- Python 3.6.3
- numpy==1.13.3
- pandas==0.20.3
- tflearn==0.3.2
- nltk==3.2.4
- tensorflow==1.4.0
- gensim==3.1.0
- scikit_learn==0.19.1

The dataset:
- The dataset contains 4743 tweets.
- There are 2413 words that appeared at least three times but are not regular English stop words (e.g. the, but).
- Preprocessing: for some words the category is more important than the actual meaning; therefore to avoid overfitting I replace all urls, @mentions, #hashtags, and numbers with 'url', '@mention', 'hashtag', and 'num', respectively

Bag-of-words model:
- Used the inverse-document-frequency term-frequency method to vectorize the tweets
- Used TensorFlow estimator LinearClassifier to perform the training
- This model is simple but is expensive to train. Training time per cycle: 1 min 50 sec.
- Five most distinct words for Donald Trump: 'dc', 'immediately', 'corrupt', 'potus', 'growing'
- Five most distinct words for Hillary Clinton:  'legislation', 'biden', 'degree', 'letting', 'duke’
- A scan of hyperparameter space showed that good results were yielded for learning_rate = 1 and l1 = 5 (parameters for the FTRL optimizer), achieving an accuracy of 86.4% (see graph below)


<img src="https://github.com/IvanChingLi/tweet-classifier/blob/master/BOW/log/hyperparam_plot.png" width="400">

Embedded word vectors:
- Used a Word2Vec model to vectorize the tweets (implemented through the gensim package)
- This model is more complicated than the bag-of-words model, but is cheaper to train. Training time per cycle: 1 min 35 sec.
- A scan of hyperparameter space showed that a good choice of embedding size is 50, which achieved an accuracy of 77.8% (see graph below)

<img src="https://github.com/IvanChingLi/tweet-classifier/blob/master/BOW_embedding/plot1.png" width="400">
