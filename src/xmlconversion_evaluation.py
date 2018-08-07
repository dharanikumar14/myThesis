import os
import sys
import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.sql.functions import *
from pyspark.ml.feature import RegexTokenizer,StopWordsRemover,CountVectorizer
from pyspark.ml.clustering import LDA,LDAModel,DistributedLDAModel
import time
from time import time 


spark = SparkSession.builder.appName('DocumentSimilarity.Rcv1.10p5.k90.20iterations.withevaluation').getOrCreate()

spark.conf.set("spark.sql.crossJoin.enabled", True)

spark.conf.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")

Start_time = time()

part = 28

custom = StructType([
		StructField("_date", StringType(), True),
		StructField("_id", LongType(), False),
		StructField("_itemid", LongType(), True),
		StructField("_lang", StringType(), True),
		StructField("byline", StringType(), True),
		StructField("copyright", StringType(), True),
		StructField("dateline", StringType(), True),
		StructField("headline", StringType(), True),
		StructField("metadata", StringType(), True),
		StructField("dc", StringType(), True),
		StructField("text", StringType(), False),
		StructField("title", StringType(), True),])

xmlDf = spark.read.format('com.databricks.spark.xml').options(rowTag='newsitem').load('hdfs://hadoop-dbse/user/pasumart/input_dataset/*',schema= custom).limit(10000)

df_first= xmlDf.select(col('_itemid').alias('DocId'),'text')



tokens = RegexTokenizer(minTokenLength=2,inputCol='text', outputCol='Words', pattern="[^a-z]+") 

Tokenized = tokens.transform(df_first)

Tokens_filtered = StopWordsRemover(inputCol='Words',outputCol='filtered_words')

Tokenized_filtered = Tokens_filtered.transform(Tokenized)


cv = CountVectorizer(inputCol="filtered_words", outputCol="features")

cv_model = cv.fit(Tokenized_filtered)  

result = cv_model.transform(Tokenized_filtered)

result1 = result.select('DocId','features').repartition(part)

#result1.show()

lda = LDA(k=90,maxIter=20,optimizer = "em")       

model = lda.fit(result1)

lda_df = model.transform(result1)

#hellinger distance metric to compare two documents

def hellinger(vec1,vec2):
    h_value = np.sqrt(0.5*((np.sqrt(vec1) - np.sqrt(vec2))**2).sum())
    return h_value.tolist()

hellinger_udf = udf( hellinger,FloatType())

def threshold(value):
    if value <= 0.2:
        return 1
    else:
        return 0

threshold_udf = udf(threshold,IntegerType())

final_df = lda_df.select('DocId','topicDistribution').repartition(part)

similarity_df = final_df.join(final_df.alias("copy_df").select(col("DocId").alias("DocId2"),col("topicDistribution").alias("topic")),\
                                 col("DocId") < col("DocId2"), 'inner')\
                                .withColumn('Score',hellinger_udf(col('topicDistribution'),col('topic')))\
                                .withColumn('Predicted_Match',threshold_udf(col('Score'))).repartition(part)


output_df = similarity_df.select('DocId','DocId2','Predicted_Match').repartition(part)


Evaluation_df = spark.read.parquet('hdfs://hadoop-dbse/user/pasumart/goldendata_10k').repartition(part)


#Total_df = output_df.join(Evaluation_df,(col("DocId") == col("DocID") )& (col("DocId2") == col("DocID2")),'inner').repartition(part)

Total_df = output_df.join(Evaluation_df,(output_df.DocId == Evaluation_df.DocID) & (output_df.DocId2 == Evaluation_df.DocID2),'inner').repartition(part)


True_Positives = Total_df.filter((Total_df['True_match']==1) & (Total_df['Predicted_Match']==1)).count()

False_Negatives = Total_df.filter((Total_df['True_match']==0) & (Total_df['Predicted_Match']==0)).count()

False_Positives =Total_df.filter((Total_df['True_match']==0) & (Total_df['Predicted_Match']==1)).count()

print("True_postivies: " +str(True_Positives))
print("False_negatives: " +str(False_Negatives))
print("False_positives: " +str(False_Positives))


precision = True_Positives/(True_Positives + False_Positives)

recall = True_Positives/(True_Positives + False_Negatives)

Fmeasure = 2*((precision*recall)/(precision + recall))

print("precision: "+str(precision))
print("Recall: "+ str(recall))
print("F-measure: "+ str(Fmeasure))


end_time = time() - Start_time
print("Total Execution time: " + str(end_time)+ " Seconds")
print("Total Execution time in minutes: " + str(end_time/60)+ " Minutes")
print("Total Execution time in hours: " + str(end_time/3600)+ " Hours")

#To calculate documents that are classified as matches

#Total_pairs = output_df.filter(output_df['Match_result']==1).count()
#print("Total document pairs matched: " + str(Total_pairs))

#output_df.select('DocId','DocId2').show()

output_df.unpersist()
Total_df.unpersist()


spark.stop()
