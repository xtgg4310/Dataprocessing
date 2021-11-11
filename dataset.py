import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, precision_score, recall_score
import joblib


def read_csvfile():
    df = pd.read_csv(r"C:\Users\Enigma_2020\Hou Haozheng Dropbox\Hou Haozheng\PC\Desktop\news.csv")
    df.columns = ["number", "title", "text", "label"]
    df.loc[:, "label"] = df["label"].apply(lambda x: 0 if x == "FAKE" else 1)
    return df


def dataset_spilt(df):
    x = df['text']
    y = df['label']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=100)
    return x_train, x_test, y_train, y_test


def Pre_pipeline(x_train, y_train):
    clf = Pipeline([  # Creating a pipeline
        ("vec", CountVectorizer(ngram_range=(1, 2))),  # The count vectorizer using default params
        ("logistic", LogisticRegression(max_iter=100000))
    ])
    clf.fit(x_train, y_train)
    return clf


def test_Text_classfication(clf, x_test, y_test):
    y_pred = clf.predict(x_test)
    print("AUC : {:.4f}".format(roc_auc_score(y_test, y_pred)))
    print("Precision: {:.4f}".format(precision_score(y_test, y_pred)))
    print("Recall : {:.4f}".format(recall_score(y_test, y_pred)))


def save_pkl(clf):
    joblib.dump(clf, "model.pkl")


if __name__ == '__main__':
    df = read_csvfile()
    x, x1, y, y1 = dataset_spilt(df)
    clf = Pre_pipeline(x, y)
    test_Text_classfication(clf, x1, y1)
    save_pkl(clf)
