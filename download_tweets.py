#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 19:56:47 2018

@author: ivan
"""

#!/usr/bin/env python
# encoding: utf-8

import tweepy #https://github.com/tweepy/tweepy
import csv
import pdb

#Twitter API credentials
consumer_key = "SY1cUrTYQWV3WPZWcue1KazCz"
consumer_secret = "rWEGJEUao3UH0nyeppgzv5UHwCp0ICLub2h0BacYXkvfHRSzTF"
access_key = "267550861-AxV89GY4JpgSxEPWDzfBR6GMlnZrgEzn09iaPPO3"
access_secret = "mv8sGwalUtqu27KD9HPLP9hiio7UkPfaZigA1v4fsdfje"

def get_all_tweets(screen_name):
#Twitter only allows access to a users most recent 3240 tweets with this method

    #authorize twitter, initialize tweepy
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_key, access_secret)
    api = tweepy.API(auth)

    #initialize a list to hold all the tweepy Tweets
    alltweets = []

    #make initial request for most recent tweets (200 is the maximum allowed count)
    new_tweets = api.user_timeline(screen_name = screen_name,count=200)

    #save most recent tweets
    alltweets.extend(new_tweets)

    #save the id of the oldest tweet less one
    oldest = alltweets[-1].id - 1 #first number is oldest id of batch 1

    #keep grabbing tweets until there are no tweets left to grab
    while len(new_tweets) > 0:
        print("getting tweets before %s" % (oldest))

        #all subsiquent requests use the max_id param to prevent duplicates
        new_tweets = api.user_timeline(screen_name = screen_name,count=200,max_id=oldest)
    
        #save most recent tweets
        alltweets.extend(new_tweets)
    
        #update the id of the oldest tweet less one
        oldest = alltweets[-1].id - 1
    
        print("...%s tweets downloaded so far" % (len(alltweets)))

    #transform the tweepy tweets into a 2D array that will populate the csv	
    outtweets = [[tweet.id_str, tweet.created_at, tweet.text.encode("utf-8")] for tweet in alltweets]
    
    #exclude retweets
    i=0
    while i <= len(outtweets)-1:
        if str(outtweets[i][2]).startswith('b\'RT'):
            outtweets.pop(i)
        else:
            i+=1
    #write the csv	
    with open('%s_tweets.csv' % screen_name, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(["id","created_at","text"])
        writer.writerows(outtweets)

pdb.set_trace()

if __name__ == '__main__':
#pass in the username of the account you want to download
    get_all_tweets("realDonaldTrump")