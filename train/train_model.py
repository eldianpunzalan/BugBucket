import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

IN = '/home/eldian/Documents/DA_BugBucket/dataset/bugbucket_clean.csv'
MODEL_OUT = '/home/eldian/Documents/DA_BugBucket/bow_model.joblib'
VEC_OUT = '/home/eldian/Documents/DA_BugBucket/bow_vectorizer.joblib'

pd.set_option('display.width', 250)
pd.set_option('display.max_columns', 20)

df = pd.read_csv(IN)
print('loaded:', len(df))

train, test = train_test_split(
    df, test_size=0.2, stratify=df['label'], random_state=42)
print('train:', len(train))
print('test:', len(test))

vectorizer = CountVectorizer(ngram_range=(1, 2))
# added accuracy 

X_train = vectorizer.fit_transform(train['clean'])
X_test = vectorizer.transform(test['clean'])
print('words (columns):', X_train.shape[1])

model = LogisticRegression(max_iter=2000, class_weight='balanced', C=3)
model.fit(X_train, train['label'])

pred = model.predict(X_test)
print('accuracy:', round(accuracy_score(test['label'], pred), 4))
print('macro F1:', round(f1_score(test['label'], pred, average='macro'), 4))

print()
print(classification_report(test['label'], pred, digits=3))

cm = confusion_matrix(test['label'], pred, labels=model.classes_)
cm_table = pd.DataFrame(cm, index=model.classes_, columns=model.classes_)
print('confusion matrix (rows = real category, columns = predicted category):')
print(cm_table)

words = vectorizer.get_feature_names_out()
print()
for i, category in enumerate(model.classes_):
    top = model.coef_[i].argsort()[::-1][:15]
    print(category + ':', ', '.join(words[j] for j in top))

joblib.dump(model, MODEL_OUT)
joblib.dump(vectorizer, VEC_OUT)
print()
print('saved:', MODEL_OUT, VEC_OUT)