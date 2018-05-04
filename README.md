# tweet-classifier
Machine learning algorithm to classify tweets as Hillary Clinton or Donald Trump. I used three methods:
1. Bag-of-words model (average accuracy: 86.5% +/- 1.8%*)
2. Embedded word vectors (average accuracy: 85.3% +/- 3.6%)
3. Recurrent neural network (average accuracy: 95.2% +/- 3.6%).

[Click here to input your own sentences into the model!](https://the-trumpest.herokuapp.com/)

*Can be thought of as 95% C.I.

<h2>Interesting Findings</h2>

1. The algorithm only needed half the tweet for most cases. 

<img src="https://github.com/IvanChingLi/tweet-classifier/blob/master/activations_visualized/to_readme/clinton2.png" width="800">
<img src="https://github.com/IvanChingLi/tweet-classifier/blob/master/activations_visualized/to_readme/clinton3.png" width="800">
<img src="https://github.com/IvanChingLi/tweet-classifier/blob/master/activations_visualized/to_readme/trump1.png" width="800">
<img src="https://github.com/IvanChingLi/tweet-classifier/blob/master/activations_visualized/to_readme/trump3.png" width="800">

2. There were some close calls.
<img src="https://github.com/IvanChingLi/tweet-classifier/blob/master/activations_visualized/to_readme/clinton1.png" width="800">
<img src="https://github.com/IvanChingLi/tweet-classifier/blob/master/activations_visualized/to_readme/trump2.png" width="800">


3. Certain words distinguished the two politicians from each other.

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

4. Although many words were used, there were roughly 20 "ideas" (embedding dimensions). Accounting for more than 20 does not improve the accuracy.

<img src="https://github.com/IvanChingLi/tweet-classifier/blob/master/BOW_embedding/embed_plot.png" width="400">

5. The algorithm generalizes poorly to tweets from other people, i.e. the algorithm learned Trump vs. Hillary, not Trump vs. not-Trump or Hillary vs. not-Hillary.
- Barack Obama (@BarackObama): 62% Hillary, 38% Trump
- Ellen DeGeneres (@TheEllenShow): 19% Hillary, 81% Trump
- Conan O'Brien (@ConanOBrien): 12% Hillary, 88% Trump

<h2>Technical Details</h2>


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
- I analyzed the tweets in the one-year period leading up to the 2016 election (Nov. 8, 2015 - Nov. 7, 2016).
- The dataset contains 9975 tweets, 5629 from Hillary Clinton and 4346 from Donald Trump.
- There were 12107 unique words.
- Preprocessing: for some words the category is more important than the actual meaning; therefore to avoid overfitting I replace all urls, @mentions, #hashtags, and numbers with 'url', '@mention', 'hashtag', and 'num', respectively

1. Bag-of-words model:
- This model is simple but is expensive to train. Training time per cycle: 1 min 50 sec.
- Used the inverse-document-frequency term-frequency method to vectorize the tweets
- Used TensorFlow estimator LinearClassifier to perform the training
- A scan of hyperparameter space showed that good results were yielded for learning_rate = 1 and l1 = 5 (parameters for the FTRL optimizer), achieving an accuracy of 86.5%

2. Embedded word vectors:
- This model is more sophisticated than the bag-of-words model, but is cheaper to train. Training time per cycle: 1 min 35 sec.
- Used a Word2Vec model to vectorize the tweets (implemented through the gensim package)
- My results show that the accuracy saturates at an embedding size of around 20 (see plot below).
- Using word embedding allows tremendous dimensionality reduction (from 2413 dimensions in the bag-of-words model to 20  dimensions) with minimal decrease in accuracy (86.5% and 85.3% are statistically indistinguishable given the uncertainties).


3. Recurrent neural network
- Used LSTM (long short term memory) cell with dropout, achieved average accuracy of 95.2%.
- Performance does not improve beyond two neurons (one layer).
- Used pre-trained weights from Word2Vec

