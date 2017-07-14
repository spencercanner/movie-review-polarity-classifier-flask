import os
import pickle

from flask import Flask, request, render_template
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn import linear_model
from sklearn .metrics import accuracy_score

app = Flask(__name__)
app.config.from_object(__name__)
app.config.from_envvar('APP_SETTINGS', silent=True)

classes = ['pos', 'neg']

# This sets this default page to the home if there is no '/'
@app.route('/')
def home():
    return render_template('home.html')


@app.route('/review', methods=['POST','GET'])
def review():
    if request.method == 'POST':
        result = request.form

        vector_file = open('vectorizer.pickle', 'rb')
        vectorizer = pickle.load(vector_file)
        review_vector = vectorizer.transform([result['review']])


        predictions = []

        model_file = open('classifier-svm.pickle', 'rb')
        classifier = pickle.load(model_file)
        prediction = classifier.predict(review_vector)
        predictions.append("Support Vector Machines: " + prediction[0])

        model_file = open('classifier-logreg.pickle', 'rb')
        classifier = pickle.load(model_file)
        prediction = classifier.predict(review_vector)
        predictions.append("Logistic Regression: " + prediction[0])

        return render_template('result.html',predictions=predictions)
    else:
        render_template('home.html')


@app.route('/train', methods=['POST', 'GET'])
def train_model():
    if request.method == 'POST':
        train_data = []
        test_data = []
        train_labels = []
        test_labels = []

        # Read in preprocessed data from project directory
        for current_class in classes:
            current_path = os.path.join('./data', current_class)
            for current_file in os.listdir(current_path):
                with open(os.path.join(current_path, current_file), 'r') as open_file:
                    content = open_file.read()
                    if current_file.startswith('cv0') or current_file.startswith('cv1'):
                        test_data.append(content)
                        test_labels.append(current_class)
                    else:
                        train_data.append(content)
                        train_labels.append(current_class)

        # Create feature vectors for the movie reviews
        vectorizer = TfidfVectorizer(min_df=5, max_df=0.8, sublinear_tf=True, use_idf=True, stop_words='english')
        train_vectors = vectorizer.fit_transform(train_data)
        test_vectors = vectorizer.transform(test_data)

        with open('vectorizer.pickle', 'wb') as fid:
            pickle.dump(vectorizer, fid, 2)

        accuracy = []

        # Create and test accuracy of Linear Support Vector Machine classifier
        classifier = svm.LinearSVC()
        classifier.fit(train_vectors, train_labels)
        prediction = classifier.predict(test_vectors)
        accuracy.append("Support Vector Machines: " + str(accuracy_score(test_labels, prediction)))

        with open('classifier-svm.pickle', 'wb') as fid:
            pickle.dump(classifier, fid, 2)

        # Create and test accuracy of Logistic Regression classifier
        classifier = linear_model.LogisticRegression()
        classifier.fit(train_vectors, train_labels)
        prediction = classifier.predict(test_vectors)
        accuracy.append("Logistic Regression: " + str(accuracy_score(test_labels, prediction)))

        with open('classifier-logreg.pickle', 'wb') as fid:
            pickle.dump(classifier, fid, 2)

        return render_template('train.html', accuracy=accuracy)
    else:
        return  render_template('home.html')
