import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn import svm
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def classifier(X_train, y_train, X_test, y_test):
    clf = svm.SVC(kernel='linear')
    #clf = HistGradientBoostingClassifier()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    cm = confusion_matrix(y_test, y_pred, labels=clf.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
    disp.plot()
    plt.show()

def main():
    vectorDF = pd.read_csv("./VectorDF.csv")
    vectorTFDF = pd.read_csv("./VectorTFDF.csv")

    X = vectorTFDF.drop('0', axis=1)
    y = vectorTFDF['0']
    y.fillna(False, inplace=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    classifier(X_train, y_train, X_test, y_test)
    
if __name__ == "__main__":
    main()