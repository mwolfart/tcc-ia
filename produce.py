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
from sklearn.ensemble import HistGradientBoostingClassifier
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

def remove_non_alpha(text):
    pattern = re.compile(r'[^a-zA-Z\s,.]')
    clean_string = re.sub(pattern, '', text)
    return clean_string

def discretize_rating(rating):
    if rating >= GOOD_RATING:
        return 1
    # elif rating >= MEDIUM_RATING:
    #     return 1
    else: return 0
    
# def linear_regression(X_train, y_train, X_test, y_test):
#     model = LinearRegression()
#     model.fit(X_train, y_train)
#     y_pred = model.predict(X_test)
    
#     errors = []
#     for i, y in enumerate(y_test):
#         error = abs(y - y_pred[i])
#         errors.append(error)
        
#     print(errors)
#     sum = 0
#     for e in errors:
#         sum += e
#     print(sum)

def main():
    nltk.download()
    df = pd.read_csv("./games.csv")
    dfu = df[df.userRatingCount > MINIMUM_RATINGS]
    dfu.reset_index(inplace=True)
    dfu['descriptionWCount'] = dfu.description.apply(lambda d : len(re.findall(r'\w+', d)))
    dfu['descriptionCCount'] = dfu.description.apply(len)
    dfu['goodRating'] = dfu.averageUserRating.apply(discretize_rating)
    # dfu['descriptionAlpha'] = dfu.description.apply(remove_non_alpha)
    
    vectorizer = CountVectorizer(analyzer=simplify_text).fit(dfu.description)
    vector = vectorizer.transform(dfu.description)
    vectorDF = pd.DataFrame(vector.todense(), columns=vectorizer.get_feature_names_out())
    vectorDF = pd.concat([dfu.goodRating, vectorDF], axis=1, ignore_index=True)
    # print("Shape of vectorized DF:", vectorDF.shape)
    # print(vectorDF.head(10))
    
    vectorizerTF = TfidfTransformer().fit(vector)
    vectorTF = vectorizerTF.transform(vector)
    vectorTFDF = pd.DataFrame(vectorTF.todense(), columns=vectorizer.get_feature_names_out())
    vectorTFDF = pd.concat([dfu.goodRating, vectorTFDF], axis=1, ignore_index=True)
    # print("Shape of vector TFDF:", vectorTFDF.shape)
    # print(vectorTFDF.head(10))
    
    vectorDF.to_csv("VectorDF.csv")
    vectorTFDF.to_csv("VectorTFDF.csv")
    
if __name__ == "__main__":
    main()