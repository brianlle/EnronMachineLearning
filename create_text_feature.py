
# coding: utf-8

# In[1]:

import sys
import pickle
from nltk.stem.snowball import SnowballStemmer
from sklearn import tree
import string
import numpy as np
from sklearn import cross_validation
from sklearn.feature_extraction.text import TfidfVectorizer


# In[2]:

def parseOutText(f):
    '''
    given an opened email file f, parse out all text below the
    metadata block at the top
        
    example use case:
    f = open("email_file_name.txt", "r")
    text = parseOutText(f)
        
    '''


    f.seek(0)  ### go back to beginning of file (annoying)
    all_text = f.read()

    ### split off metadata
    content = all_text.split("X-FileName:")

    words = ""
    if len(content) > 1:
        ### remove punctuation
        text_string = content[1].translate(string.maketrans("", ""), string.punctuation)

        ### split the text string into individual words, stem each word,
        ### and append the stemmed word to words (make sure there's a single
        ### space between each stemmed word)
        
        wordList = text_string.split()
        for word in wordList:

            #word = word.replace(' ', '')
            stemmer = SnowballStemmer("english")
            stemmedWord = stemmer.stem(word)

            if len(stemmedWord) > 1:
                words = words + (stemmedWord) + ' '
        
        words = words[:-1]



    return words


# In[3]:

def predict_emails(email, word_data, author_data, vectorizer, classifier):
    
    '''
    Once decision tree classifier is created, predict_emails uses it to classify
    an email as being either from a person of interest (poi) or not. Email is used
    to identify locations in author_data and pull up corresponding already
    stemmed emails from word_data
    
    Returns list of two ints: number of emails predicted to be sent by poi,
                              and total number of emails read
    '''
    
    poi_predictions = 0
    total_emails = 0
    
    person_data = [i for i,j in zip(word_data, author_data) if j == email]
    if len(person_data) > 0:
        person_data_vectorized  = vectorizer.transform(person_data).toarray()
        person_data_predicted = clf.predict(person_data_vectorized)
    
        poi_predictions = person_data_predicted.sum()
        total_emails = len(person_data_predicted)
    
    return [poi_predictions, total_emails]


# In[4]:

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)


# In[5]:

poiEmails = []

for person in data_dict:
    if data_dict[person]['poi'] == True:
        poiEmails.append(data_dict[person]['email_address'])


# In[6]:

notPoiEmails = []

for person in data_dict:
    if data_dict[person]['poi'] == False:
        if data_dict[person]['email_address'] != 'NaN':
            notPoiEmails.append(data_dict[person]['email_address'])


# In[7]:

poiEmailPaths = []
notPoiEmailPaths = []
author_data = []

for email in poiEmails:
    path = 'emails_by_address/from_' + email + '.txt'
    try:
        from_poi = open(path,'r')
        for email_path in from_poi:
            try:
                poiEmailPaths.append(email_path[20:-2]) # format path for access use
                author_data.append(email)
            except IOError:
                continue
        from_poi.close()
    except IOError:
        continue
    
    
for email in notPoiEmails:
    path = 'emails_by_address/from_' + email + '.txt'
    try:
        not_from_poi = open(path,'r')
        for email_path in not_from_poi:
            try:
                notPoiEmailPaths.append(email_path[20:-1]) # format path for access use
                author_data.append(email)
            except IOError:
                continue
        not_from_poi.close()
    except IOError:
        continue
    


# In[8]:

poi_data = []
word_data = []

temp_counter = 0

for poi_status, from_person in [(True, poiEmailPaths), (False, notPoiEmailPaths)]:
    for path in from_person:

        path = '..\\' + path

        email = open(path, "r")

        ### use parseOutText to extract the text from the opened email
        words = parseOutText(email)

        ### append the text to word_data
        word_data.append(words)
        poi_data.append(poi_status) #mark if from poi or not from poi
            
            
            email.close()


# In[25]:

replace_list = ['ddelainnsf', 'delaineyhouect','delainey','ect','eea','lavoratocorpenronenron','ena','david','dave']

modified_word_data = []

for word in word_data:
    for replace_word in replace_list:
        temp_word = word.replace(replace_word, '')
    modified_word_data.append(temp_word.replace('  ', ' '))  #replace double spaces with single space


# In[26]:

features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(modified_word_data, poi_data, test_size=0.1, random_state=42)

vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,
                             stop_words='english')
features_train = vectorizer.fit_transform(features_train)
features_test  = vectorizer.transform(features_test).toarray()


# In[27]:

clf = tree.DecisionTreeClassifier()
clf.fit(features_train, labels_train)


# In[24]:

email_prediction_data = {}

for email in poiEmails:
    email_prediction_data[email] = predict_emails(email, modified_word_data, author_data, vectorizer, clf)

for email in notPoiEmails:
    email_prediction_data[email] = predict_emails(email, modified_word_data, author_data, vectorizer, clf)


# In[ ]:

pickle.dump( email_prediction_data, open("your_author_prediction_data.pkl", "w") )
pickle.dump( word_data, open("your_word_data.pkl", "w") )
pickle.dump( modified_word_data, open("your_modified_word_data.pkl", "w") )
pickle.dump( poi_data, open("your_poi_status.pkl", "w") )
pickle.dump( author_data, open("your_author_data.pkl", "w") )

