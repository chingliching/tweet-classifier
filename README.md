# tweet-classifier
Machine learning algorithm to classify tweets as Hillary Clinton or Donald Trump. I used three methods:
1. Bag-of-words model (average accuracy: 86.5% +/- 1.8%*, training time per cycle: 1 min 50 sec)
2. Embedded word vectors (average accuracy: 85.3% +/- 3.6%, training time per cycle: 1 min 35 sec)
3. Recurrent neutral network (average accuracy: 91.3% +/- 1.9%).

*Can be thought of as 95% C.I.

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

1. Bag-of-words model:
- This model is simple but is expensive to train. Training time per cycle: 1 min 50 sec.
- Used the inverse-document-frequency term-frequency method to vectorize the tweets
- Used TensorFlow estimator LinearClassifier to perform the training
- A scan of hyperparameter space showed that good results were yielded for learning_rate = 1 and l1 = 5 (parameters for the FTRL optimizer), achieving an accuracy of 86.5%

| Top Trumpian Words  | Top Clintonian Words |
| ------------- | ------------- |
| D.C.  | reflects   |
| immediately  | Duke   |
| corrupt  | letting   |
| POTUS  | degree   |
| growing  | Biden  |
| protesters  | legislation  |
| chairman  | facts  |
| pols  | officials  |
| Crooked  | nominate  |
| center  | progressive  |



2. Embedded word vectors:
- This model is more sophisticated than the bag-of-words model, but is cheaper to train. Training time per cycle: 1 min 35 sec.
- Used a Word2Vec model to vectorize the tweets (implemented through the gensim package)
- My results show that the accuracy saturates at an embedding size of around 20 (see plot below).
- Using word embedding allows tremendous dimensionality reduction (from 2413 dimensions in the bag-of-words model to 20  dimensions) with minimal decrease in accuracy (86.4% and 85.8% are statistically indistinguishable given the uncertainties).

<img src="https://github.com/IvanChingLi/tweet-classifier/blob/master/BOW_embedding/embed_plot.png" width="400">

3. Recurrent neural network
- Use LSTM (long short term memory) cell with dropout

