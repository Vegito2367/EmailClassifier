import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from flask import Flask,request
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay


app = Flask(__name__)
class EmailClassifier:
    def __init__(self):
        fig,ax=plt.subplots()
        self.logModel=LogisticRegression()

        dataframe=pd.read_csv("EmailClassifier/email_text.csv")

        output=dataframe['label']
        textemail=dataframe['text']

        

        X_train, X_test, Y_train, Y_test = train_test_split(textemail, output, test_size=0.3, random_state=3)

        self.extractor = TfidfVectorizer(min_df=1, stop_words="english",lowercase=True)

        X_train_features = self.extractor.fit_transform(X_train)
        X_test_features = self.extractor.transform(X_test)
        
        Names="Spam","Ham"
        sizes=[int(output.value_counts()[1]),int(output.value_counts()[0])]
        ax.pie(sizes,labels=Names,autopct='%1.1f%%')

        # Names="Test Data","Training Data"
        # sizes=[int(X_test_features.shape[0]),int(X_train_features.shape[0])]
        # ax.pie(sizes,labels=Names,autopct='%1.1f%%')

        titles_options = [
            ("Confusion matrix for spam/ham, without normalization", None),
        ]
        for title, normalize in titles_options:
            disp = ConfusionMatrixDisplay.from_estimator(
                self.logModel,
                X_test_features,
                Y_test,
                display_labels=["spam","ham"],
                cmap=plt.cm.Blues,
                normalize=normalize,
            )
            disp.ax_.set_title(title)
            print(title)
            print(disp.confusion_matrix)
            plt.show()
        print(X_train_features)
        
        
        
        Y_test=Y_test.astype('int')
        Y_train=Y_train.astype('int')

        self.logModel.fit(X_train_features,Y_train)
        prediction_test=self.logModel.predict(X_test_features)
        accuracy=accuracy_score(Y_test,prediction_test)

        print(f" Accuracy on test data = {accuracy}")
        plt.show()


    def classify(self,email):
        inputdata=self.extractor.transform(email)

        prediction = self.logModel.predict(inputdata)

        return prediction




