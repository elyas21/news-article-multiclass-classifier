# %%


# %%
# !unzip archive.zip

import numpy as np 
import pandas as pd 
import nltk
import string as s
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
# %matplotlib inline
import seaborn as sns
import re
import os



# %%

nltk.download('stopwords')

# %%


train_data=pd.read_csv("data/train.csv",header=2,names=['classid','title','desc'])
test_data=pd.read_csv("data/test.csv",header=0,names=['classid','title','desc'])

train_data.head()

test_data.head()

train_data.shape

test_data.shape



# %%
# sns.countplot(train_data.classid)

# %%


sns.countplot(test_data.classid)

# %%


train_x=train_data.desc
test_x=test_data.desc
train_y=train_data.classid
test_y=test_data.classid



# %%

def remove_html(text):
    html_pattern = re.compile('<.*?>')
    return html_pattern.sub(r'', text)

train_x=train_x.apply(remove_html)
test_x=test_x.apply(remove_html)



# %%
def remove_urls(text):
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)





# %%
train_x=train_x.apply(remove_urls)
test_x=test_x.apply(remove_urls)

# %%

def word_tokenize(txt):
    tokens = re.findall("[\w']+", txt)
    return tokens



# %%
train_x=train_x.apply(word_tokenize)
test_x=test_x.apply(word_tokenize)

# %%

def remove_stopwords(lst):
    stop=stopwords.words('english')
    new_lst=[]
    for i in lst:
        if i.lower() not in stop:
            new_lst.append(i)
    return new_lst


# %%

train_x=train_x.apply(remove_stopwords)
test_x=test_x.apply(remove_stopwords) 


# %%

def remove_punctuations(lst):
    new_lst=[]
    for i in lst:
        for  j in  s.punctuation:
            i=i.replace(j,'')
        new_lst.append(i)
    return new_lst


# %%
train_x=train_x.apply(remove_punctuations) 
test_x=test_x.apply(remove_punctuations)



# %%
def remove_numbers(lst):
    nodig_lst=[]
    new_lst=[]

    for i in  lst:
        for j in  s.digits:
            i=i.replace(j,'')
        nodig_lst.append(i)
    for i in  nodig_lst:
        if  i!='':
            new_lst.append(i)

    return new_lst


# %%
train_x=train_x.apply(remove_numbers)
test_x=test_x.apply(remove_numbers)

import nltk

# %%

def stemming(text):
    porter_stemmer = nltk.PorterStemmer()
    roots = [porter_stemmer.stem(each) for each in text]
    return (roots)

# %%


train_x=train_x.apply(stemming)
test_x=test_x.apply(stemming)

# %%

lemmatizer=nltk.stem.WordNetLemmatizer()


# %%

def lemmatzation(lst):
    new_lst=[]
    for i in lst:
        i=lemmatizer.lemmatize(i)
        new_lst.append(i)
    return new_lst


# %%
nltk.download('wordnet')

# %%
train_x=train_x.apply(lemmatzation)
test_x=test_x.apply(lemmatzation)

# %%

def remove_extrawords(lst):
    stop=['href','lt','gt','ii','iii','ie','quot','com']
    new_lst=[]
    for i in lst:
        if i not in stop:
            new_lst.append(i)
    return new_lst



# %%
train_x=train_x.apply(remove_extrawords)
test_x=test_x.apply(remove_extrawords) 

# %%
train_x=train_x.apply(lambda x: ''.join(i+' ' for i in x))
test_x=test_x.apply(lambda x: ''.join(i+' '  for i in x))

# %%
import sklearn
print(sklearn.__version__)

# %%
from sklearn.feature_extraction.text import TfidfVectorizer

# Initialize TfidfVectorizer with specific parameters
tfidf = TfidfVectorizer(min_df=8, ngram_range=(1, 3))

# Fit and transform the train data
train_1 = tfidf.fit_transform(train_x)

# Transform the test data using the trained vectorizer
test_1 = tfidf.transform(test_x)




# %%
# Print the number of features extracted
print("No. of features extracted")
print(len(tfidf.get_feature_names_out()))  # Use get_feature_names_out() in version 1.6.1

# Print the first 100 feature names
print(tfidf.get_feature_names_out()[:10])

# %%
print(train_1.shape)
print(test_1.shape)

# %%
# You can pass these directly to many scikit-learn models, like:
from sklearn.linear_model import LogisticRegression



# %%
model = LogisticRegression()
model.fit(train_1, train_y)


# %%

# Test on the sparse test set
predictions = model.predict(test_1)

# %%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
import time  # Import time module

# %%


# Use sparse matrix instead of converting to dense array
pd.DataFrame(train_1[:100].toarray(), columns=tfidf.get_feature_names_out())



# %%
# Start timing
start_time = time.time()

# Training the Naive Bayes model
NB_MN = MultinomialNB(alpha=0.52)
NB_MN.fit(train_1, train_y)  # Use train_1 directly as a sparse matrix
pred = NB_MN.predict(test_1)  # Use test_1 directly as a sparse matrix

# %%


# First 20 actual and predicted labels
print("First 20 actual labels")
print(test_y.tolist()[:20])
print("First 20 predicted labels")
print(pred.tolist()[:20])

# %%


# F1 score and Accuracy evaluation
print("F1 score of the model")
print(f1_score(test_y, pred, average='micro'))
print("Accuracy of the model")
print(accuracy_score(test_y, pred))
print("Accuracy of the model in percentage")
print(f"{round(accuracy_score(test_y, pred) * 100, 3)} %")



# %%

# Confusion Matrix and Visualization
sns.set(font_scale=1.5)
cof = confusion_matrix(test_y, pred)

# %%


# Convert confusion matrix into a DataFrame for better readability
cof = pd.DataFrame(cof, index=[i for i in range(1, 5)], columns=[i for i in range(1, 5)])



# %%
# Plotting the confusion matrix
plt.figure(figsize=(8, 8))
sns.heatmap(cof, cmap="PuRd", linewidths=1, annot=True, square=True, cbar=False, fmt='d',
            xticklabels=['World', 'Sports', 'Business', 'Science'],
            yticklabels=['World', 'Sports', 'Business', 'Science'])
plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")
plt.title("Confusion Matrix for News Article Classification")
plt.show()

# %%


# End timing
end_time = time.time()
print(f"Training and evaluation took {end_time - start_time} seconds")

# %%


# %%



