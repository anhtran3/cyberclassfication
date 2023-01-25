# %%
import pandas as pd
import numpy as np
from numpy.random import RandomState
from wordcloud import WordCloud

# %%
# Load the original dataset
data = pd.read_csv('data/cyberbullying_tweets.csv')


# %%
data.shape

# %%
# Rename columns for simplicity
data = data.rename(
    columns = {'tweet_text': 'text', 'cyberbullying_type':'type'}
)

# %%
rng = RandomState(1)
df = data.sample(n = len(data)-100, random_state = rng)


# %%
test_data = data.loc[~data.index.isin(df.index)]


# %% [markdown]
# # Data exploration 
# 

# %%
df.info()

# %%
df.sample(5)

# %%
# Check null values
np.sum(df.isnull())

# %%
print(df['type'].value_counts())

# %%
import matplotlib.pyplot as plt
y_axis = df['type'].value_counts()
x_axis = df['type'].unique()

colors = ['green', 'blue', 'purple', 'brown', 'teal', "orange"]
plt.barh(x_axis,y_axis, color = colors)
plt.show()

# %% [markdown]
# ## DATA PREPROCESSING 

# %% [markdown]
# 
# There are not much imbalance among different types remove other_cyberbullying type since it may cause confusion for the models with other cyberbullying class
# 

# %%

df = df.drop(index = df[(df['type'] == "other_cyberbullying")].index)

# %%
from sklearn.preprocessing import LabelEncoder
lenc = LabelEncoder()

df['sentiment'] = lenc.fit_transform(df['type'])

# %%
df[["type", "sentiment"]].value_counts()

# %%
sentiments = ["age", "ethnicity", "gender", "not bullying", "religion"]

# %%
df.info()

# %% [markdown]
# # Text cleaning to remove noise

# %%
import regex as re
import string
import nltk
import contractions
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer,PorterStemmer
stop_words = set(stopwords.words('english'))

# %%
# remove contractions
def decontract(text):
    return contractions.fix(text)

# %%
# remove '#, _' symbol from words 
def clean_hashtags(tweet):
    new_tweet= " ".join(word.strip() for word in re.split('#|_', tweet)) 
    return new_tweet

# %%
# remove punctuation, links, stopwords, mentions
def strip_all_entities(text):
    text = text.replace('\r', '').replace('\n', ' ')
    text = text.lower()
    # remove links and mentions
    text = re.sub(r"(\@|https?\://|(www.))\S+", "", text)
    # remove non utf8/ascii characters
    text = re.sub(r'[^\x00-\x7f]',r'', text) 
    # remove punctuation
    table = str.maketrans('', '',string.punctuation)
    text = text.translate(table)
    # remove stopwords
    text = [
        word for word in text.split() if word not in stop_words
    ]
    text = ' '.join(text)
    text =' '.join(word for word in text.split() if len(word) < 14)
    # remove multiple sequential spaces
    text = re.sub(r' +', " ", text)
    # remove digits
    text = re.sub(r"\d", "", text)
    return text

# %%
# Stemming
def stemmer(text):
    tokenized = nltk.word_tokenize(text)
    ps = PorterStemmer()
    return ' '.join([ps.stem(words) for words in tokenized])

# %%
# Lemmatization 
def lemmatize(text):
    tokenized = nltk.word_tokenize(text)
    lm = WordNetLemmatizer()
    return ' '.join([lm.lemmatize(words) for words in tokenized])

# %%
# Combine all functions to preprocess
def preprocess(text):
    text = decontract(text)
    text = strip_all_entities(text)
    text = clean_hashtags(text)
    text = lemmatize(text)
    return text

# %%
texts_cleaned = []
for t in df['text']:
    texts_cleaned.append(preprocess(t))


# %%
df["text_clean"] = texts_cleaned

# %%
df["text_clean"].duplicated().sum()

# %%
# remove duplicates
df.drop_duplicates("text_clean", inplace=True)

# %%
df["sentiment"].value_counts()

# %% [markdown]
# **After removing duplicates, the value counts per sentiment is shown above. </br>
# There is no major imbalance in the preprocessed data**

# %%
text_len = []
for text in df["text_clean"]:
    tweet_len = len(text.split())
    text_len.append(tweet_len)

# %%
df['text_len'] = text_len

# %%
import seaborn as sns
plt.figure(figsize=(10,5))
ax = sns.countplot(x='text_len', data=df[df['text_len']<20], palette='mako')
plt.title('Count of tweets with less than 20 words', fontsize=20)
plt.yticks([])
ax.bar_label(ax.containers[0])
plt.ylabel('count')
plt.xlabel('')
plt.show()

# %%
df["text_len"].value_counts().sort_values(ascending= False).head()

# %%
# checking long tweets
df.sort_values(by=["text_len"], ascending=False)

# %% [markdown]
# **Removing tweets with less than 4 words and more than 50 words as they can be outliers**

# %%
df = df[df['text_len'] > 3]
df = df[df['text_len'] < 50]

# %%
df.shape

# %% [markdown]
# # Use WordCloud to visualize

# %%
# plotting word cloud excluding not_cyberbullying
new_df = df
new_df = new_df[new_df['type'] != 'not_cyberbullying']
new_df = new_df['text'].apply(lambda x: "".join(x))

# %%
plt.figure(figsize= (20,20))
wc = WordCloud(max_words=1000, width= 1600, height= 800, 
                collocations= False).generate(''.join(new_df))
plt.imshow(wc)

# %% [markdown]
# # Vectorize text into numbers

# %%
from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn import metrics
from sklearn.metrics import classification_report

cv = CountVectorizer()
X_cv =  cv.fit_transform(df['text_clean'])

tf_transformer = TfidfTransformer(use_idf=True).fit(X_cv)
X_tf = tf_transformer.transform(X_cv)

# %%
X_tf

# %% [markdown]
# # Split into train and validation data
# 

# %%
from sklearn.model_selection import train_test_split
from sklearn import metrics
X_train, X_test, y_train, y_test = train_test_split(
    X_tf, df['sentiment'], test_size=0.20, stratify=df['sentiment'], random_state=42
)

# %%
y_train.value_counts()

# %% [markdown]
# There are imbalance in cyberbullying types, therefore, we apply SMOTE technique to balance the data

# %% [markdown]
# # SMOTE

# %%
from imblearn.over_sampling import SMOTE
vc = y_train.value_counts()
while (vc[0] != vc[4]) or (vc[0] !=  vc[2]) or (vc[0] !=  vc[3]) or (vc[0] !=  vc[1]):
    smote = SMOTE(sampling_strategy='minority')
    X_train, y_train = smote.fit_resample(X_train, y_train)
    vc = y_train.value_counts()
vc

# %% [markdown]
# # Create models
# 

# %%
# Random forest
from sklearn.ensemble import RandomForestClassifier
rf_clf = RandomForestClassifier()
rf_clf.fit(X_train, y_train)
rf_pred = rf_clf.predict(X_test)
print('Classification Report for Random Forest:\n',classification_report(y_test, rf_pred, target_names=sentiments))

# %%
# Naives Bayes
from sklearn.naive_bayes import MultinomialNB
nb_clf = MultinomialNB()
nb_clf.fit(X_train, y_train)
nb_pred = nb_clf.predict(X_test)
print('Classification Report for Naive Bayes:\n',classification_report(y_test, nb_pred, target_names=sentiments))


# %%
# SVM
from sklearn.svm import SVC
svm_clf = SVC(kernel= 'linear', C = 2).fit(X_train, y_train)
svm_pred = svm_clf.predict(X_test)
print('Classification Report for SVM:\n',classification_report(y_test, svm_pred, target_names=sentiments))


# %% [markdown]
# ## CROSS VALIDATION

# %%
from sklearn.model_selection import cross_val_score

RF_cv_score = cross_val_score(rf_clf,X_train, y_train, cv=2)
SVM_cv_score = cross_val_score(svm_clf,X_train, y_train, cv=2)

print('Cross validation score (Random Forest Classifier):', RF_cv_score.mean())
print('Cross validation score (Support Vectors Machine):', SVM_cv_score.mean())



# %% [markdown]
# The CV score are similar to the test accuracy; therefore, we did not overfit nor underfit the model

# %%
# Create a confusion matrix using heatmap
from sklearn.metrics import confusion_matrix 
def print_confusion_matrix(
    confusion_matrix, class_names, figsize = (8,5), fontsize=10
):

    df = pd.DataFrame(
        confusion_matrix, index=class_names, columns=class_names, 
    )
    fig = plt.figure(figsize=figsize)
    try:
        heatmap = sns.heatmap(df, annot=True, fmt="d")
    except ValueError:
        raise ValueError("Confusion matrix values must be integers.")
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize)
    plt.ylabel('Truth')
    plt.xlabel('Prediction')

# %%
# Random Forest
cm = confusion_matrix(y_test,rf_pred)
print_confusion_matrix(cm,sentiments)

# %% [markdown]
# Most number of gender tweets are classified as not_bullying. Perhaps it is due to the oversampling SMOTE technique on gender and not_bullying observations
# 

# %%
# Naive Bayes
cm = confusion_matrix(y_test,nb_pred)
print_confusion_matrix(cm,sentiments)

# %%
# SVM
cm = confusion_matrix(y_test,svm_pred)
print_confusion_matrix(cm,sentiments)

# %% [markdown]
# # Apply to the new test data

# %% [markdown]
# Clean test data by dropping 'other_cyberllying' sentiment and apply preprocessing 

# %%
test_data = test_data.drop(index = test_data[(test_data['type'] == "other_cyberbullying")].index)
test_data["sentiment"] = lenc.fit_transform(test_data["type"])

# %%
test_data.shape

# %%
# Cleaning text test set
import string
test_texts_cleaned = []
for t in test_data.text:
    test_texts_cleaned.append(preprocess(t))


# %%
test_cv = cv.transform(test_texts_cleaned)
tf_transformer = TfidfTransformer(use_idf=True).fit(test_cv)
X_tf_test = tf_transformer.transform(test_cv)

# %%
# Apply random forest on the new test data 
rf_predictions = rf_clf.predict(X_tf_test)
# RF accuracy
print("Test accuracy:", metrics.accuracy_score(test_data["sentiment"], rf_predictions))

# %% [markdown]
# sentiments = ["age", "ethnicity", "gender", "not bullying", "religion"]

# %%
rf_predictions

# %%
rf_sentiments = lenc.inverse_transform(rf_predictions)

# %%
test_data.drop('sentiment',axis=1, inplace= True)
test_data['rf_prediction'] = rf_sentiments

# %% [markdown]
# # Sample prediction of RF on the test data

# %%
test_data

# %%
cm = confusion_matrix(test_data["type"],test_data["rf_prediction"])
print_confusion_matrix(cm,sentiments)

# %% [markdown]
# # Demonstrate classfication on random tweets

# %%
def prep_convert(t):
    text = preprocess(t)
    cv_text = cv.transform([text])
    tf_transformer = TfidfTransformer(use_idf=True).fit(cv_text)
    X_tf_test = tf_transformer.transform(cv_text)
    return X_tf_test

# %%
example = "why are you Christian in highschool?"
result = rf_clf.predict(prep_convert(example))
lenc.inverse_transform(result)


