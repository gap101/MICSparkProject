from pyspark.sql import SparkSession
from pyspark.ml.regression import LinearRegression


spark = SparkSession.builder.appName("generatedApplicaton").getOrCreate() 
a9c760080_844f_f74b_5211_558e00bfeda2 = spark.read.format("libsvm").load('D:\Spark\spark-2.2.0-bin-hadoop2.7\spark-2.2.0-bin-hadoop2.7\data\mllib\sample_linear_regression_data.txt')
LinearRegressor = LinearRegression(featuresCol = "features", labelCol = "label", predictionCol = "prediction", maxIter= 100, regParam= 0, elasticNetParam = 0 ,standardization = True, tol = 0.000001, fitIntercept = True, solver = "auto", weightCol = None, aggregationDepth = 2)
  
a72372596_33d2_d7c9_31b8_13fd143d138b = LinearRegressor.fit(a9c760080_844f_f74b_5211_558e00bfeda2)

print("Summary : %s " % str(a72372596_33d2_d7c9_31b8_13fd143d138b)) 
