import os
import pickle
import matplotlib.pyplot as plt

from flask import Flask, request, render_template, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm
from sklearn import linear_model
from sklearn .metrics import accuracy_score
import numpy as np

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
        plot = plot_review_coefficients(classifier, review_vector, vectorizer.get_feature_names())
        plot.savefig('./app/static/images/review-svm.png', transparent='true', bbox_inches='tight', pad_inches=0)
        predictions.append(["Support Vector Machines", format_prediction(prediction[0]),
                            "./static/images/review-svm.png"])

        model_file = open('classifier-logreg.pickle', 'rb')
        classifier = pickle.load(model_file)
        prediction = classifier.predict(review_vector)
        plot = plot_review_coefficients(classifier, review_vector, vectorizer.get_feature_names())
        plot.savefig('./app/static/images/review-logreg.png', transparent='true', bbox_inches='tight', pad_inches=0)
        predictions.append(["Logistic Regression", format_prediction(prediction[0]),
                            "./static/images/review-logreg.png"])

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
        plot = plot_coefficients(classifier, vectorizer.get_feature_names())
        plot.savefig('./app/static/images/svm.png', transparent='true', bbox_inches='tight', pad_inches=0)
        accuracy.append(["Support Vector Machines", str(accuracy_score(test_labels, prediction)),
                         './static/images/svm.png'])


        with open('classifier-svm.pickle', 'wb') as fid:
            pickle.dump(classifier, fid, 2)

        # Create and test accuracy of Logistic Regression classifier
        classifier = linear_model.LogisticRegression()
        classifier.fit(train_vectors, train_labels)
        prediction = classifier.predict(test_vectors)
        plot = plot_coefficients(classifier, vectorizer.get_feature_names())
        plot.savefig('./app/static/images/logreg.png', bbox_inches='tight', pad_inches=0)
        accuracy.append(["Logistic Regression", str(accuracy_score(test_labels, prediction)),
                         './static/images/logreg.png'])

        with open('classifier-logreg.pickle', 'wb') as fid:
            pickle.dump(classifier, fid, 2)

        return render_template('train.html', accuracy=accuracy)
    else:
        return render_template('home.html')


def format_prediction(prediction):
    if prediction == 'pos':
        return "That was a positive review."
    else:
        return "That was a negative review."


def plot_coefficients(classifier, feature_names, top_features=10):
    coef = classifier.coef_.ravel()
    top_positive_coefficients = np.argsort(coef)[-top_features:]
    top_negative_coefficients = np.argsort(coef)[:top_features]
    top_coefficients = np.hstack([top_negative_coefficients, top_positive_coefficients])
    plt.figure(figsize=(9, 4))
    colors = ["red" if c < 0 else "green" for c in coef[top_coefficients]]
    plt.bar(np.arange(2 * top_features), coef[top_coefficients], color=colors)
    feature_names = np.array(feature_names)
    plt.xticks(np.arange(0, 2 * top_features), feature_names[top_coefficients], rotation=60, ha="right")
    plt.tight_layout()
    return plt


def plot_review_coefficients(classifier, review_vector, feature_names ):
    top_features = int(len(review_vector.indices)) if len(review_vector.indices) < 20 else 20
    coef = classifier.coef_.ravel()
    review_coef = [coef[index] for index in review_vector.indices]
    review_feature_names = [feature_names[index] for index in review_vector.indices]
    top_coefficients = np.argsort(np.absolute(review_coef))[-top_features:]
    review_coef = np.array(review_coef)
    review_feature_names = np.array(review_feature_names)
    plt.figure(figsize=(9, 4))
    colors = ["red" if c < 0 else "green" for c in review_coef[top_coefficients]]
    plt.bar(np.arange(top_features), review_coef[top_coefficients], color=colors)
    plt.xticks(np.arange(0, top_features), review_feature_names[top_coefficients], rotation=60, ha="right")
    plt.tight_layout()
    return plt
