from flask import Flask, request, render_template, redirect, session
from redditHandler import fetch_posts
from model_prediction import predict_sentiment, tokenizer
import base64
from io import BytesIO
import matplotlib.pyplot as plt

# https://stackoverflow.com/questions/27611216/how-to-pass-a-variable-between-flask-pages global variables

app = Flask(__name__)
app.config['SECRET_KEY'] = 'sentimentAnalysisKey'

def sentiment_count_helper(sentiment):
     sentiment_count = {"positive": 0, "negative": 0}
     
     for senti in sentiment:
          if senti == 1:
               sentiment_count["positive"] += 1
          else:
               sentiment_count["negative"] += 1
     
     return sentiment_count

def incr_word_count(text, word_count):
     if text in word_count:
          word_count[text] += 1
     else:
          word_count[text] = 1

def top_pos_neg_word_posts(titles, sentiment):
     pos_word_count = {}
     neg_word_count = {}
     
     for idx in range(len(titles)):
          text_list  = tokenizer(titles[idx])
          senti_pred = sentiment[idx]
          
          for text in text_list:
               text = text.lower()
               
               if senti_pred == 1:
                    incr_word_count(text, pos_word_count)
               elif senti_pred == 0:
                    incr_word_count(text, neg_word_count)

     sorted_pos_word_count = dict(sorted(pos_word_count.items(), key=lambda item: item[1]))
     sorted_neg_word_count = dict(sorted(neg_word_count.items(), key=lambda item: item[1]))
     
     top_five_pos_word = list(sorted_pos_word_count.keys())[-5:]
     top_five_neg_word = list(sorted_neg_word_count.keys())[-5:]

     return top_five_pos_word, top_five_neg_word

@app.route('/')
def index():
     return render_template('homePage.html')

@app.route('/searchSubreddit', methods=["POST"])
def lookUpSubreddit():
     if 'Subreddit' not in request.form:
          return redirect('/')

     session["subreddit"] = request.form['Subreddit']

     return redirect('/subreddit')

@app.route('/subreddit', methods=["GET"])
def sentimentAnlysisPage():
     subreddit = session.get('subreddit', None)
     titles, scores = fetch_posts(subreddit)
     
     sentiment = []
     for title in titles:
          pred_senti = predict_sentiment(title)
          sentiment.append(pred_senti)

     sentiment_count = sentiment_count_helper(sentiment)
     labels = list(sentiment_count.keys())
     count  = list(sentiment_count.values())
     plt.bar(labels, count)
     plt.xlabel('Sentiment')
     plt.ylabel('Count')
     plt.title('Sentiment Count of Subreddit Posts')


     buf = BytesIO()
     plt.savefig(buf, format='png')
     plt.close()
     buf.seek(0)

     sentiment_plot = base64.b64encode(buf.getbuffer()).decode("ascii")

     top_five_pos_word, top_five_neg_word = top_pos_neg_word_posts(titles, sentiment)
     print(top_five_pos_word)
     
     return render_template('analysisPage.html', subreddit=subreddit, titles=titles, 
               upvotes=scores, sentiment=sentiment, length=len(titles),
               sentiment_plot=sentiment_plot, top_five_pos_word=top_five_pos_word,
               top_five_neg_word = top_five_neg_word)

if __name__ == "__main__":
    app.run(debug=True)