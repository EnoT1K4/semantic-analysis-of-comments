from pyspark.sql import SparkSession
from pyspark.sql.functions import udf
from pyspark.sql.types import StringType, ArrayType, DoubleType
from pyspark.sql import functions as F
from pyspark.sql.functions import col, udf, array
from pyspark.ml.feature import RegexTokenizer
from pyspark.ml.classification import LogisticRegression
import pyspark.sql.functions as F
import pyspark.ml.feature as feature
import pyspark.ml.classification as classification
from pyspark.ml.feature import  HashingTF, IDF, StringIndexer
from pyspark.ml.evaluation import BinaryClassificationEvaluator

import pandas as pd
from pymystem3 import Mystem
import pymorphy2
import numpy as np
import re
import time
import nltk
from nltk.corpus import stopwords as nltk_stopwords

from plotly.graph_objects import Figure, Violin, Layout
import plotly.express as px

spark = SparkSession.builder.appName("spark").getOrCreate()

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

# Load the data
positive = spark.read.csv('/home/vladislav/kursed/positive.csv', sep=',', header=None)
negative = spark.read.csv('/home/vladislav/kursed/negative.csv', sep=';', header=None)
start_clean = time.time()
positive_text = positive.select('_c3').toDF('text')
negative_text = negative.select('_c3').toDF('text')
positive_text = positive_text.withColumn('label', F.lit(1))
negative_text = negative_text.withColumn('label', F.lit(0))
labeled_tweets = positive_text.union(negative_text)

# Define the stopwords

stopwords_udf = udf(clean_stop_words, StringType())
clear_text_udf = udf(clear_text, StringType())

labeled_tweets = labeled_tweets.filter(labeled_tweets.text.isNotNull())
labeled_tweets = labeled_tweets.withColumn('text_clear', clear_text_udf(labeled_tweets.text))
labeled_tweets = labeled_tweets.withColumn('text_clear', stopwords_udf(col('text_clear')))

# Load the second dataset
labeled_texts_1 = spark.read.csv('/home/vladislav/kursed/text_rating_final.csv',header=None)
labeled_texts_1 = labeled_texts_1.select('_c0', '_c1').toDF('text', 'label')
labeled_texts_1 = labeled_texts_1.withColumn('label', F.col('label').cast('double'))
labeled_texts_1 = labeled_texts_1.filter(F.col('label')!= 0)
labeled_texts_1 = labeled_texts_1.withColumn('label_binary', F.when(F.col('label') < 0, 0).otherwise(1))
labeled_texts_1 = labeled_texts_1.select('text', 'label_binary',).toDF('text', 'label')

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
start_clean = time.time()
data = spark.read.csv("/home/vladislav/kursed/youtube_comments_data.csv", header=True, inferSchema=True)
data = data.select('comment').toDF('text')


data = data.filter(data.text.isNotNull())
data = data.withColumn('text_clear', clear_text_udf(data.text))
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
'''
share_neg = []
for i in range(output_dataset.count()):

    if output_dataset.select("toxicity").collect()[i][0][1] > 0.45:
        share_neg.append(output_dataset.select("toxicity").collect()[0][0][1])


point_of = len(share_neg) / output_dataset.count()
print('Share of negative comments: '+str(round(share_neg, 2)))



fig = Figure(
    output_dataset=Violin(
        y=output_dataset.select("toxicity").toPandas()["toxicity"],
        meanline_visible=True,
        name="(N = %i)" % output_dataset.count(),
        side="positive",
        spanmode="hard"
    ),
    layout=Layout(
        height=500,
        xaxis_showgrid=False,
        xaxis_zeroline=False,
        template="plotly_dark",
        font_color="rgba(212, 210, 210, 1)",
        legend=dict(
            y=0.9,
            x=-0.1,
            yanchor="top",
        ),
    )
)
fig.add_annotation(x=0.8, y=1.5,
            text = "%0.2f — доля негативных комментариев (при p > 0.44)"\
                   % share_neg,
            showarrow=False,
            yshift=10)

fig.update_traces(orientation="h", width=1.5, points=False)
fig.update_yaxes(visible=False)

fig.show()
'''