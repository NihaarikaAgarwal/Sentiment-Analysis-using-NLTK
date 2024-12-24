from flask import Flask, request, send_file , render_template
import os
from os import listdir
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from collections import Counter
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from wordcloud import WordCloud
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('agg')
import numpy as np
from PIL import Image

app = Flask(__name__ , template_folder="template")

picfolder = os.path.join('static','images')
app.config['UPLOAD_FOLDER']= picfolder

def clean(text):
#cleaning the text (converting to lower case and removing punctuation)
 lower_case = text.lower()
 cleaned_text = lower_case.translate(str.maketrans('', '', string.punctuation))
 return(cleaned_text)

def pre_proscessing(cleaned_text):
 # tokenizing words
 tokenized_words = word_tokenize(cleaned_text, "english")
 #print(tokenized_words)
 # Removing Stop Words
 final_words = []
 for word in tokenized_words:
     if word not in stopwords.words('english'):
         final_words.append(word)
 #print(final_words)
 # Lemmatization - From plural to single + Base form of a word (example better-> good)
 lemma_words = []
 for word in final_words:
     word = WordNetLemmatizer().lemmatize(word)
     lemma_words.append(word)
 emotion_list = []
 emotion_words = []
 with open('emotions.txt', 'r') as file:
     for line in file:
         clear_line = line.replace("\n", '').replace(",", '').replace("'", '').strip()
         word, emotion = clear_line.split(':')
         if word in lemma_words:
             emotion_words.append(word)
             emotion_list.append(emotion)
 w = Counter(emotion_list)
 content_words = ' '.join(emotion_words)
 return(w,content_words)

def sentiment_analysis(cleaned_text):
    score = SentimentIntensityAnalyzer().polarity_scores(cleaned_text)
    if score['pos'] > score['neg']:
        feeling = "Positive"
        value = score['pos'] * 100
    elif score['neg'] > score['pos']:
        feeling = "Negative"
        value = score['neg'] * 100
    else:
        feeling = ("Neutral")
    return (feeling , value)

@app.route('/home')
def home():
    title = os.path.join(app.config['UPLOAD_FOLDER'], 'head.jpeg')
    return render_template('page1.html',fig=title)

@app.route('/')
def my_form():
    return render_template('page3.html')

@app.route('/', methods=['POST'])
def my_form_post():
    text = request.form['text']
    with open('read.txt', 'w') as f:
        f.write(str(text))
    texts = open("read.txt").read()
    cleaned_text = clean(texts)
    feel, val = sentiment_analysis(cleaned_text)
    return render_template("page4.html",feeling=feel,value=val)

@app.route('/plotting')
def plotting():
     text = open("read.txt").read()
     cleaned_text = clean(text)
     w, content_words = pre_proscessing(cleaned_text)
     # plotting the words v/s frequency graph
     fig, ax1 = plt.subplots()
     ax1.bar(w.keys(), w.values())
     fig.autofmt_xdate()
     ax1.set_title('Emotion V/S Frquency')
     plt.savefig('graph.png')
     return send_file(filename_or_fp='graph.png', mimetype='image/png')

@app.route('/clouds')
def clouds():
     text = open("read.txt").read()
     cleaned_text = clean(text)
     w , content_words = pre_proscessing(cleaned_text)
     cloud_mask = np.array(Image.open('cloud.png'))
     wordcloud = WordCloud(mask=cloud_mask,
                               background_color='black', contour_width=3, contour_color='lightpink', collocations=False
                               ).generate(content_words)
     plt.figure(figsize=(8,8))
     plt.imshow(wordcloud)
     plt.axis("off")
     plt.savefig('word cloud.png')
     return send_file(filename_or_fp='word cloud.png', mimetype='image/png')

@app.route('/canalysis')
def canalysis():

    text = open("covid.txt" , encoding="utf-8").read()
    cleaned_text = clean(text)
    feeling, value = sentiment_analysis(cleaned_text)
    #word cloud
    w, content_words = pre_proscessing(cleaned_text)
    cloud_mask = np.array(Image.open('cloud.png'))
    wordcloud = WordCloud(mask=cloud_mask,
                          background_color='black', contour_width=3, contour_color='lightpink', collocations=False
                          ).generate(content_words)
    plt.figure(figsize=(8, 8))
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.savefig('ccloud.png')
    #graph
    fig, ax1 = plt.subplots()
    ax1.bar(w.keys(), w.values())
    fig.autofmt_xdate()
    ax1.set_title('Emotion V/S Frquency')
    plt.savefig('cgraph.png')
    ccloud = os.path.join(app.config['UPLOAD_FOLDER'], 'ccloud4.png')
    cgraph = os.path.join(app.config['UPLOAD_FOLDER'], 'cgraph.png')
    return render_template("2page.html",feeling=feeling,value=value,fig1=ccloud,fig2=cgraph)

@app.route('/cplotting')
def cplotting():
     text = open("covid.txt").read()
     cleaned_text = clean(text)
     w, content_words = pre_proscessing(cleaned_text)
     # plotting the words v/s frequency graph
     fig, ax1 = plt.subplots()
     ax1.bar(w.keys(), w.values())
     fig.autofmt_xdate()
     ax1.set_title('Emotion V/S Frquency')
     plt.savefig('cgraph.png')
     #return content_words
     return send_file(filename_or_fp='cgraph.png', mimetype='image/png')

@app.route('/cclouds')
def cclouds():
     text = open("covid.txt").read()
     cleaned_text = clean(text)
     w , content_words = pre_proscessing(cleaned_text)
     cloud_mask = np.array(Image.open('cloud.png'))
     wordcloud = WordCloud(mask=cloud_mask,
                               background_color='black', contour_width=3, contour_color='lightpink', collocations=False
                               ).generate(content_words)
     plt.figure(figsize=(8,8))
     plt.imshow(wordcloud)
     plt.axis("off")
     plt.savefig('ccloud.png')
     return send_file(filename_or_fp='static/images/cccloud.png', mimetype='image/png')

if __name__ == '__main__':
    app.debug = True
    use_reloader = True
    app.run()
