# News Article topic prediction.

This projects aim to build NLP model to predict the news article topic based on the heading and body of the article. 
Dataset is collection of Huffman news articles. The dataset contains around 50K news articles. All the articles are labeled based on their topic. 
We have a total of 30 different topics. 
Intially, we only have text data which ofcourse ML models cannot use. So we have transfrom the data in to numbers to build models upon that. 
Removed all the articles which had missing pieces of information. clubbed the heading and body to make a single text column. 
Processed text data to normalize it. Removed stopwords, made everything to lower case and lemmatized the text. 
Converted these word into TF-IDF vectors. This resulted in total of 30K features. Since there may be words that were used rarely we can ignore them and decrease the number of 
features. Used vectorizer to remove rare words and most common words and reduced the number of features to 5k. 
Built Logit, Naive Bayes, and SVM models to predict the topic. 

This is the updated codebase
