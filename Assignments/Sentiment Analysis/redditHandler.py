import praw
import pandas as pd
from model_prediction import predict_sentiment

user_agent = "Test API 1"

# instances of Reddit
reddit = praw.Reddit(
    client_id = "KUivSj_mwwz7tZe9-tR8eA",
    client_secret = "Bl4xX0C1ad0h2RzngPvzCjgThveCyg",
    user_agent = user_agent
    #check_for_async=False
)
# subreddit to extract posts from
def fetch_posts(subreddit_name):
    subreddit = reddit.subreddit(subreddit_name)

    titles = []
    scores = []

    for submission in subreddit.top(limit=50):
        titles.append(submission.title)
        scores.append(submission.score)
    
    return titles, scores