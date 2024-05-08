from collections import defaultdict
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from skmultilearn.problem_transform import BinaryRelevance
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer

def run_nb(train_slices, ds_dict):

    mlb = MultiLabelBinarizer()

    macro_scores, micro_scores = defaultdict(list), defaultdict(list)

    for train_slice in train_slices:
        
        # train_slices is a list with indeces.
        ds_train_sample = ds_dict['train'].select(train_slice)
        y_train = np.array(ds_train_sample['label_ids'])						# These are the 0/1 mapping vectors. 

        y_test = np.array(ds_dict['test']['label_ids'])

        # https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html
        count_vectorizer = CountVectorizer()

        X_train_counts = count_vectorizer.fit_transform(ds_train_sample['text'])
        # Type is <class 'scipy.sparse._csr.csr_matrix'>.

        X_test_counts = count_vectorizer.transform(ds_dict['test']['text'])

        classifier = BinaryRelevance(classifier=MultinomialNB())
    
        classifier.fit(X_train_counts, y_train)

        y_pred_test = classifier.predict(X_test_counts)

        clf_report = classification_report(y_test, y_pred_test, target_names=mlb.classes, zero_division=0, output_dict=True)

        macro_scores['Naive Bayes'].append(clf_report['macro avg']['f1-score'])
        micro_scores['Naive Bayes'].append(clf_report['micro avg']['f1-score'])

    return micro_scores, macro_scores

