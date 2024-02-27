import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import re
import string

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import svm
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

st = PorterStemmer()
MINIMUM_RATINGS = 30

GOOD_RATING = 4.5
MEDIUM_RATING = 4

# stemming and stop words removal
def simplify_text(text):
    simple = [char for char in text if char not in string.punctuation]
    simple = ''.join(simple).split()
    simple = [x for x in simple if x not in stopwords.words('english')]
    simple = [st.stem(x) for x in simple]
    return simple

def discretize_rating(rating):
    if rating >= GOOD_RATING:
        return 1
    # elif rating >= MEDIUM_RATING:
    #     return 1
    else: return 0
    
def linear_regression(X_train, y_train, X_test, y_test):
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    errors = []
    for i, y in enumerate(y_test):
        error = abs(y - y_pred[i])
        errors.append(error)
        
    print(errors)
    sum = 0
    for e in errors:
        sum += e
    print(sum)

def classifier(X_train, y_train, X_test, y_test):
    clf = svm.SVC(kernel='linear')
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    cm = confusion_matrix(y_test, y_pred, labels=clf.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
    disp.plot()
    plt.show()

def main():
    nltk.download()
    df = pd.read_csv("./games.csv")
    df = df[df.userRatingCount > MINIMUM_RATINGS]
    df['descriptionWCount'] = df.description.apply(lambda d : len(re.findall(r'\w+', d)))
    df['descriptionCCount'] = df.description.apply(len)
    df['goodRating'] = df.averageUserRating.apply(discretize_rating)
    # df['descriptionSimple'] = df.description.apply(simplify_text)
    
    vectorizer = CountVectorizer(analyzer=simplify_text).fit(df.description)
    vector = vectorizer.transform(df.description)
    vectorDF = pd.DataFrame(vector.todense(), columns=vectorizer.get_feature_names_out())
    vectorDF = pd.concat([df.goodRating, vectorDF], axis=1)
    print("Shape of vectorized DF:", vectorDF.shape)
    print(vectorDF.head(10))
    
    vectorizerTF = TfidfTransformer().fit(vector)
    vectorTF = vectorizerTF.transform(vector)
    vectorTFDF = pd.DataFrame(vectorTF.todense(), columns=vectorizer.get_feature_names_out())
    vectorTFDF = pd.concat([df.goodRating, vectorTFDF], axis=1)
    print("Shape of vector TFDF:", vectorTFDF.shape)
    print(vectorTFDF.head(10))
    
    X = vectorTFDF.drop('goodRating', axis=1)
    y = vectorTFDF.goodRating
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    
    classifier(X_train, X_test, y_train, y_test)
    
if __name__ == "__main__":
    main()