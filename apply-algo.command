#! /bin/zsh
. /$ZSH/oh-my-zsh.sh 
handle="rainnwilson"
filename=$handle".csv"
python2 ../GetOldTweets-python-master/Exporter.py --username $handle --maxtweets 100 --toptweets --output $filename
python tweet_filter.py $filename
rm $filename
mv $handle"_f.csv" $filename
python endpoint.py $filename