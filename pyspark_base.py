from pyspark.sql import SparkSession
from pyspark.sql.functions import udf
from pyspark.sql.types import StringType, ArrayType, DoubleType
from pyspark.sql import functions as F
from pyspark.sql.functions import col, udf, array
from pyspark.ml.feature import RegexTokenizer
from pyspark.ml.classification import LogisticRegression
import pyspark.ml.feature as feature
import pyspark.ml.classification as classification
from pyspark.ml.feature import  HashingTF, IDF, StringIndexer
from pyspark.ml.evaluation import BinaryClassificationEvaluator


import plotly.graph_objects as go
import pandas as pd
import pymorphy2
import numpy as np
import re
import time
import nltk
from nltk.corpus import stopwords as nltk_stopwords


spark = SparkSession.builder \
   .appName("Cassandra Connector Example") \
   .config("spark.jars.packages", "com.datastax.spark:spark-cassandra-connector_2.12:3.5.0") \
   .getOrCreate()
#3.5.1 4.1.4
stopwords = set(nltk_stopwords.words('russian'))

# Define the clear_text function
def clear_text(text):
    if text is None:
        return ''
    else:
        return re.sub(r'[^А-яё]+', ' ', text).lower()

# Define the clean_stop_words function
def clean_stop_words(text: str, stopwords = stopwords) -> str:
    text = [word for word in text.split() if word not in stopwords]
    return " ".join(text)
print('Start model Preprocessing')
# Load the data
#positive = spark.read.csv('/home/vladislav/kursed/positive.csv', sep=',', header=None)
#negative = spark.read.csv('/home/vladislav/kursed/negative.csv', sep=';', header=None)
start_clean = time.time()
#positive_text = positive.select('_c3').toDF('text')
#negative_text = negative.select('_c3').toDF('text')
#positive_text = positive_text.withColumn('label', F.lit(1))
#negative_text = negative_text.withColumn('label', F.lit(0))
#labeled_tweets = positive_text.union(negative_text)
#beled_tweets = labeled_tweets.na.drop()
#labeled_tweets = labeled_tweets.na.fill({"text": ""})

#labeled_tweets.write.mode("append").format("org.apache.spark.sql.cassandra").option("spark.cassandra.connection.host", "127.0.0.1").option("keyspace", "kursed").option("table", "labeled_tweets").save()
# Define the stopwords
labeled_tweets = spark.read.format("org.apache.spark.sql.cassandra") \
   .options(table="labeled_tweets", keyspace="kursed") \
   .load()
stopwords_udf = udf(clean_stop_words, StringType())
clear_text_udf = udf(clear_text, StringType())

labeled_tweets = labeled_tweets.filter(labeled_tweets.text.isNotNull())
labeled_tweets = labeled_tweets.withColumn('text_clear', clear_text_udf(labeled_tweets.text))
labeled_tweets = labeled_tweets.withColumn('text_clear', stopwords_udf(col('text_clear')))

# Load the second dataset
#labeled_texts_1 = spark.read.csv('/home/vladislav/kursed/text_rating_final.csv',header=None)
#labeled_texts_1 = labeled_texts_1.select('_c0', '_c1').toDF('text', 'label')
#labeled_texts_1 = labeled_texts_1.withColumn('label', F.col('label').cast('double'))
#labeled_texts_1 = labeled_texts_1.filter(F.col('label')!= 0)
#labeled_texts_1 = labeled_texts_1.withColumn('label_binary', F.when(F.col('label') < 0, 0).otherwise(1))
#labeled_texts_1 = labeled_texts_1.select('text', 'label_binary',).toDF('text', 'label')

#labeled_texts_1 = labeled_texts_1.na.drop()
#labeled_texts_1 = labeled_texts_1.na.fill({"text": ""})

#labeled_texts_1.write.mode("append").format("org.apache.spark.sql.cassandra").option("spark.cassandra.connection.host", "127.0.0.1").option("keyspace", "kursed").option("table", "labeled_texts_1").save()
labeled_texts_1 = spark.read.format("org.apache.spark.sql.cassandra") \
   .options(table="labeled_texts_1", keyspace="kursed") \
   .load()

# Apply the clear_text and clean_stop_words functions
labeled_texts_1 = labeled_texts_1.withColumn('text_clear', clear_text_udf(col('text')))
labeled_texts_1 = labeled_texts_1.withColumn('text_clear', stopwords_udf(col('text_clear')))

# Define the lemmatize function

regexTokenizer = RegexTokenizer(inputCol="text_clear", outputCol="tokens", pattern=r"\s+")
labeled_tweets = regexTokenizer.transform(labeled_tweets)
labeled_texts_1 = regexTokenizer.transform(labeled_texts_1)
labeled_texts_1.dropna()
labeled_tweets.dropna()
morph = pymorphy2.MorphAnalyzer()
digits = [str(i) for i in range(10)]
def lemmatize(tokens):
    return [morph.normal_forms(word)[0] for word in tokens if (word[0] not in digits and word not in stopwords)]

lemmatize_udf = F.udf(lemmatize, ArrayType(StringType()))


labeled_tweets = labeled_tweets.withColumn('lemm_text_clear', lemmatize_udf('tokens'))

# Apply the lemmatize function

labeled_texts = labeled_texts_1.withColumn('lemm_text_clear', lemmatize_udf('tokens'))

# Concatenate the two datasets
join_result = labeled_texts.union(labeled_tweets)

# split the dataset into training and testing sets
train, test = join_result.randomSplit(weights=[0.7, 0.3], seed=100)




hashingTF = HashingTF(inputCol="lemm_text_clear", outputCol="rawFeatures", numFeatures=2048)
train_features = hashingTF.transform(train)
test_features = hashingTF.transform(test)
idf = IDF(inputCol="rawFeatures", outputCol="features")
idfModel = idf.fit(train_features)
train_features = idfModel.transform(train_features)
test_features = idfModel.transform(test_features)

indexer = StringIndexer(inputCol="label", outputCol="indexedLabel")
indexerModel = indexer.fit(train)
train_indexed = indexerModel.transform(train_features)
test_indexed = indexerModel.transform(test_features)


lr = LogisticRegression(featuresCol="features", labelCol="indexedLabel", maxIter=10000, regParam=0.01)
lrModel = lr.fit(train_indexed)
#lrModel.save("./model","overwrite")
print('Model Preprocessing time: '+str(round(time.time() - start_clean, 2))+' seconds')
print('Start model Learning')
start_clean = time.time()
data = spark.read.format("org.apache.spark.sql.cassandra") \
   .options(table="comments", keyspace="kursed") \
   .load()
#data = spark.read.csv("/home/vladislav/kursed/youtube_comments_data.csv", header=True, inferSchema=True)

data = data.select('comment').toDF('text')
data = data.withColumn('text_clear', clear_text_udf(col('text')))
data = data.withColumn('text_clear', stopwords_udf(col('text_clear')))
data = regexTokenizer.transform(data)
data = data.withColumn('lemm_text_clear', lemmatize_udf('tokens'))


hashingTF = HashingTF(inputCol="lemm_text_clear", outputCol="rawFeatures", numFeatures=2048)
output_test = hashingTF.transform(data)

idf1 = IDF(inputCol="rawFeatures", outputCol="features")
idfModel1 = idf1.fit(output_test)
dataset = idfModel1.transform(output_test)

predictions = lrModel.transform(dataset)


output_dataset = predictions.select('probability').toDF('toxicity')
print('Model Learning time: '+str(round(time.time() - start_clean, 2))+' seconds')

output_dataset_l = output_dataset.collect()

# Write the list to a text file
with open("output.txt", "w") as f:
    for row in output_dataset_l:
        f.write(str(row) + "\n")
        