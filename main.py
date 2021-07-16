#!/usr/bin/env python
# coding: utf-8

# In[18]:


import nltk
from nltk.corpus import stopwords
import pandas as pd
default_stopwords = stopwords.words('english')
import seaborn as sns
sns.set_style("darkgrid")
import warnings
warnings.filterwarnings('ignore')
import string
from nltk.stem import PorterStemmer, WordNetLemmatizer


# In[19]:


all_data = pd.read_csv('Huff_news.csv', parse_dates = True, index_col = 0)

all_data.dropna(inplace = True)

all_data["text_all"] = all_data["headline"] +" "+ all_data["short_description"]


# In[4]:


def clean_text(doc,
            rm_punctuation=True,
            rm_digits=True,
            lemmatize=True,
            norm_case=True,
            stem=False,
            rm_stopwords=True):

            # Doc overall operations
            if(rm_digits==True):
                table = str.maketrans({key: None for key in string.digits})
                doc = str(doc).translate(table)
            if(norm_case==True):
                doc = doc.lower()
            if(rm_punctuation==True):
                table = str.maketrans({key: None for key in string.punctuation})
                doc = doc.translate(table)
            if(rm_stopwords==True):
                words = " ".join([i for i in doc.split() if i not in default_stopwords])
            else:
                words = " ".join([i for i in doc.split()])
            if(lemmatize==True):
                lemma = WordNetLemmatizer()
                words = " ".join(lemma.lemmatize(word) for word in words.split())
            if(stem==True):
                porter_stemmer = PorterStemmer()
                words = " ".join(porter_stemmer.stem(word) for word in words.split())
            return words

# In[5]:


all_data["document"] = [clean_text(x,lemmatize=True) for x in all_data.text_all]


# In[6]:

X, y =all_data.document, all_data.category


# In[7]:


from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(max_features=3800, max_df=0.9, min_df=0.005, stop_words=stopwords.words('english'))
x = vectorizer.fit_transform(all_data.document).toarray()


# In[8]:


#tf idf
from sklearn.feature_extraction.text import TfidfTransformer
tfidfconverter = TfidfTransformer()
x = tfidfconverter.fit_transform(x).toarray()

# In[9]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)


# In[10]:


from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB


# In[11]:


from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score


# In[12]:


#Logit model
logistic = LogisticRegression()
logistic.fit(X_train, y_train) 

# test - prediction

logistic_prediction = logistic.predict(X_test)


#accuracy score
print(accuracy_score(logistic_prediction, y_test))


#confusion matrix
logit_confusionmatrix=confusion_matrix(logistic_prediction, y_test)



#classification report
print(classification_report(logistic_prediction, y_test))


# In[13]:


from sklearn.metrics import precision_recall_fscore_support as score
p,r,f1,s=score(y_test,logistic_prediction )
a=pd.Series(f1).sort_values(ascending=False).head(3).index.to_list()
all_labels=y.unique()
all_labels.sort()
list(all_labels[a])
# Top 3 categories predicted from logit model with respect to the f1 score


# In[14]:


from yellowbrick.classifier import ClassificationReport
from yellowbrick.classifier import ConfusionMatrix


# In[15]:


def evaluate_model(model):
     model.fit(X_train, y_train)
     cr = ClassificationReport(model)
     cr.score(X_test, y_test)
     cr.finalize()


# In[16]:


evaluate_model(LogisticRegression())


# In[ ]:


# Naive Bayes Classifier

# Naive Bayes 


nb =  GaussianNB()
nb.fit(X_train, y_train) 

nb_prediction = nb.predict(X_test)



print(accuracy_score(nb_prediction, y_test))


nb_confusionmatrix=confusion_matrix(nb_prediction, y_test)

#classification report
print(classification_report(nb_prediction, y_test))


# In[ ]:


p_nb,r_nb,f1_nb,s_nb=score(y_test,nb_prediction )
b=pd.Series(f1_nb).sort_values(ascending=False).head(3).index.to_list()
all_labels=y.unique()
all_labels.sort()
list(all_labels[b])
# Top 3 predictions from NB classifier


# In[ ]:


evaluate_model(GaussianNB())


# In[ ]:


from sklearn.svm import SVC

SVC_model = SVC()

SVC_model.fit(X_train, y_train)

SVC_prediction = SVC_model.predict(X_test)


print(accuracy_score(SVC_prediction, y_test))


svc_confusionmatix=confusion_matrix(SVC_prediction, y_test)


print(classification_report(SVC_prediction, y_test))


# In[ ]:


evaluate_model(SVC())


# In[ ]:


p_svc,r_svc,f1_svc,s_svc=score(y_test,SVC_prediction )
c=pd.Series(f1_svc).sort_values(ascending=False).head(3).index.to_list()
all_labels=y.unique()
all_labels.sort()
list(all_labels[c])
# Top 3 predictions from SVC classifier


# In[ ]:
