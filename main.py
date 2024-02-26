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
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

st = PorterStemmer()

# stemming and stop words removal
def simplify_text(text):
    simple = [char for char in text if char not in string.punctuation]
    simple = ''.join(simple).split()
    simple = [x for x in simple if x not in stopwords.words('english')]
    simple = [st.stem(x) for x in simple]
    return simple

def main():
    nltk.download()
    df = pd.read_csv("./games.csv")
    df['description_wcount'] = df.description.apply(lambda d : len(re.findall(r'\w+', d)))
    df['description_ccount'] = df.description.apply(len)
    # df['description_simple'] = df.description.apply(simplify_text)
    
    vectorizer = CountVectorizer(analyzer=simplify_text).fit(df.description)
    vector = vectorizer.transform(df.description)
    vectorDF = pd.DataFrame(vector.todense(), columns=vectorizer.get_feature_names_out())
    vectorDF = pd.concat([df.averageUserRating, vectorDF], axis=1)
    print("Shape of vectorized DF:", vectorDF.shape)
    print(vectorDF.head(10))
    
    vectorizerTF = TfidfTransformer().fit(vector)
    vectorTF = vectorizerTF.transform(vector)
    vectorTFDF = pd.DataFrame(vectorTF.todense(), columns=vectorizer.get_feature_names_out())
    vectorTFDF = pd.concat([df.averageUserRating, vectorTFDF], axis=1)
    print("Shape of vector TFDF:", vectorTFDF.shape)
    print(vectorTFDF.head(10))
    
    X = vectorTFDF.drop('averageUserRating', axis=1)
    y = vectorTFDF.averageUserRating
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.show()
    
if __name__ == "__main__":
    main()