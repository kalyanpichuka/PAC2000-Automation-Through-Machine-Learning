from flask import Flask,render_template, request,json
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression
import pandas as pd


app = Flask(__name__)

dataset = pd.read_csv('dataset - enlarged.csv', skiprows=1, header=None)
vectorizer = CountVectorizer()
train_bow_set = vectorizer.fit_transform(dataset[0])
tfidf_transformer = TfidfTransformer().fit(train_bow_set)
messages_tfidf = tfidf_transformer.transform(train_bow_set)
detect_model = LogisticRegression().fit(messages_tfidf,dataset[1])


@app.route('/')
def home():
	return render_template('homePage.html')
		
@app.route('/result',methods = ['POST', 'GET'])
def result():
	if request.method == 'POST':
		result = request.form['Resolution']
		bow = vectorizer.transform([result])
		tfidf4 = tfidf_transformer.transform(bow)
		resolution = detect_model.predict(tfidf4)
		return render_template("result.html",resolution = resolution)
	  
if __name__=="__main__":
    app.run(debug=True)