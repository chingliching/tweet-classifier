# tweet-classifier
Machine learning algorithm to classify tweets as Hillary Clinton or Donald Trump. I used three methods:
1. Bag-of-words model (average accuracy: 86.4%)
2. Embedded word vectors (average accuracy: 77.8%)
3. Recurrent neutral network (in progress).

Preprocessing:
- For some words the category is more important than the actual meaning; therefore to avoid overfitting I replace all urls, @mentions, #hashtags, and numbers with 'url', '@mention', 'hashtag', and 'num', respectively

Bag-of-words model:
- Used the inverse-document-frequency term-frequency method to vectorize the tweets
- Used TensorFlow estimator LinearClassifier to perform the training
- This model is simple but is expensive to train.
- A scan of hyperparameter space showed that good results were yielded for learning_rate = 1 and l1 = 5 (parameters for the FTRL optimizer), achieving an accuracy of 86.4%.

Embedded word vectors:
- Used a Word2Vec model to vectorize the tweets (implemented through the gensim package)
- This model is more involved than the bag-of-words model, but is cheaper to train.
- A scan of hyperparameter space showed that a good choice of embedding size is 50, which achieved an accuracy of 77.8%

[[https://github.com/IvanChingLi/tweet-classifier/blob/master/BOW_embedding/plot1.png]]
![alt text](tweet-classifier/BOW_embedding/plot1.png)
![My image](/BOW_embedding/tweet-classifier/BOW_embedding/plot1.png)
![My image](tweet-classifier/BOW_embedding/plot1.png)
![My image](/BOW_embedding/plot1.png)
