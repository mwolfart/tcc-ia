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
# MINIMUM_RATINGS = 30
GOOD_RATING = 4.6
THRESHOLD_NON_ZERO_COLS = 10
USE_TFDF = True
USE_RELEASE_NOTES = False
HIGH_ENGAGEMENT_THRESHOLD = 20000
NUM_ITERATIONS = 20

GET_TOP_WORDS = False
# MEDIUM_RATING = 4


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


def classifier(X_train, y_train, X_test, y_test):
    # clf = svm.SVC(kernel='linear')
    clf = HistGradientBoostingClassifier()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    prec = tp / (tp + fp)
    rec = tp / (tp + fn)
    acc = (tp+tn) / (tp + tn + fp + fn)
    return tn, fn, fp, tp, acc, rec, prec
    # disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
    # disp.plot()
    # plt.show()

def main():
    # nltk.download()
    df = pd.read_csv("./games.csv")
    # dfu = df[df.userRatingCount > MINIMUM_RATINGS]
    dfu = df
    dfu.reset_index(inplace=True)
    dfu['descriptionWCount'] = dfu.description.apply(lambda d : len(re.findall(r'\w+', d)))
    dfu['descriptionCCount'] = dfu.description.apply(len)
    dfu['goodRating'] = dfu.averageUserRating.apply(discretize_rating)
    dfu['highEngagement'] = dfu.userRatingCount.apply(lambda x: x > HIGH_ENGAGEMENT_THRESHOLD)
    # dfu['descriptionAlpha'] = dfu.description.apply(remove_non_alpha)
    
    if USE_RELEASE_NOTES:
        for i, r in dfu.iterrows():
            if not pd.isna(r['releaseNotes']):
                df.at[i, 'description'] = r['description'] + "\n" + r['releaseNotes']
    
    if GET_TOP_WORDS:
        dfuHigh = dfu[dfu.highEngagement == True]
        dfuLow = dfu[dfu.highEngagement == False]
        
        vectorizer = CountVectorizer(analyzer=simplify_text).fit(dfuHigh.description)
        vector = vectorizer.transform(dfuHigh.description)
        vectorizerTF = TfidfTransformer().fit(vector)
        vectorTF = vectorizerTF.transform(vector)
        vectorTFDF = pd.DataFrame(vectorTF.todense(), columns=vectorizer.get_feature_names_out())
        
        highWords = vectorTFDF.sum()
        
        vectorizer = CountVectorizer(analyzer=simplify_text).fit(dfuLow.description)
        vector = vectorizer.transform(dfuLow.description)
        vectorizerTF = TfidfTransformer().fit(vector)
        vectorTF = vectorizerTF.transform(vector)
        vectorTFDF = pd.DataFrame(vectorTF.todense(), columns=vectorizer.get_feature_names_out())
        
        lowWords = vectorTFDF.sum()
        result = highWords.copy()
        result.update(highWords.sub(lowWords, fill_value=0))
        # result = highWords - lowWords
        
        # print(highWords.sort_values().head(10))
        # print(lowWords.sort_values().head(10))
        # print(result.sort_values().head(10))
        # print(highWords.sort_values().tail(10))
        # print(lowWords.sort_values().tail(10))
        # print(result.sort_values().tail(10))
        
        print("Words that provide high engagement")
        print(result.sort_values().tail(30))
        
        return
    
    vectorizer = CountVectorizer(analyzer=simplify_text).fit(dfu.description)
    vector = vectorizer.transform(dfu.description)
    vectorDF = pd.DataFrame(vector.todense(), columns=vectorizer.get_feature_names_out())
    vectorDF = pd.concat([dfu.highEngagement, vectorDF], axis=1, ignore_index=True)
    # print("Shape of vectorized DF:", vectorDF.shape)
    # print(vectorDF.head(10))
    
    vectorizerTF = TfidfTransformer().fit(vector)
    vectorTF = vectorizerTF.transform(vector)
    vectorTFDF = pd.DataFrame(vectorTF.todense(), columns=vectorizer.get_feature_names_out())
    vectorTFDF = pd.concat([dfu.highEngagement, vectorTFDF], axis=1, ignore_index=True)
    # print("Shape of vector TFDF:", vectorTFDF.shape)
    # print(vectorTFDF.head(10))
    
    # vectorDF.to_csv("VectorDF.csv")
    # vectorTFDF.to_csv("VectorTFDF.csv")
    
    vec = vectorTFDF if USE_TFDF else vectorDF
    
    non_zero_counts = vec.astype(bool).sum(axis=1)
    columns_to_drop = non_zero_counts[non_zero_counts < THRESHOLD_NON_ZERO_COLS].index
    vec = vec.drop(columns_to_drop, axis=1)
    
    # vectorTFDF.to_csv("temp.csv")
    X = vec.drop(0, axis=1)
    y = vec[0]
    # y.fillna(False, inplace=True)
    
    sumAcc = 0
    sumPrec = 0
    sumRec = 0
    
    sumTn = 0
    sumFn = 0
    sumTp = 0
    sumFp = 0
    
    for i in range(NUM_ITERATIONS):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        tn, fn, fp, tp, acc, rec, prec = classifier(X_train, y_train, X_test, y_test)
        print(tn, fn, fp, tp, acc, rec, prec)
        sumTn += tn
        sumFn += fn
        sumFp += fp
        sumTp += tp
        sumAcc += acc
        sumPrec += prec
        sumRec += rec
    
    print("avg accuracy:", sumAcc / NUM_ITERATIONS)
    print("avg precision:", sumPrec / NUM_ITERATIONS)
    print("avg recall:", sumRec / NUM_ITERATIONS)
    print("avg tn:", sumTn / NUM_ITERATIONS)
    print("avg fn:", sumFn / NUM_ITERATIONS)
    print("avg tp:", sumTp / NUM_ITERATIONS)
    print("avg fp:", sumFp / NUM_ITERATIONS)
    
if __name__ == "__main__":
    main()