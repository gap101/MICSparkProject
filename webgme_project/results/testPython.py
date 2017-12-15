from pyspark.sql import SparkSession
from pyspark.ml.evaluation import RegressionEvaluator

from pyspark.ml.regression import LinearRegression


spark = SparkSession.builder.appName("generatedApplicaton").getOrCreate() 
aaf90be7b_6e16_95ce_4ac4_03f32382b74b = spark.read.format("libsvm").load('D:\Spark\spark-2.2.0-bin-hadoop2.7\spark-2.2.0-bin-hadoop2.7\data\mllib\sample_linear_regression_data.txt')
(a3fe96886_2c9c_ccdf_6761_738f2e33f551, a6b3e1bd9_cc5f_ebf5_b80b_69cf582cfd98) = aaf90be7b_6e16_95ce_4ac4_03f32382b74b.randomSplit([0.8, 0.2])

LinearRegressor = LinearRegression(featuresCol = "features", labelCol = "label", predictionCol = "prediction", maxIter= 100, regParam= 0, elasticNetParam = 0 ,standardization = True, tol = 0.000001, fitIntercept = True, solver = "auto", aggregationDepth = 2)
  
acf14b2c1_bcec_c133_d388_b0a0d75c94f5 = LinearRegressor.fit(a3fe96886_2c9c_ccdf_6761_738f2e33f551)

print("Coefficients: %s" % str(acf14b2c1_bcec_c133_d388_b0a0d75c94f5.coefficients))
print("Intercept: %s" % str(acf14b2c1_bcec_c133_d388_b0a0d75c94f5.intercept))
a3606a6f2_c32e_a469_e642_78cd23403450 = acf14b2c1_bcec_c133_d388_b0a0d75c94f5.transform(a6b3e1bd9_cc5f_ebf5_b80b_69cf582cfd98)
evaluator = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="rmse")
a9f47c719_ac97_2ad0_4007_d2c7c1f2954f = evaluator.evaluate(a3606a6f2_c32e_a469_e642_78cd23403450)
print("Root Mean Squared Error : " + str(a9f47c719_ac97_2ad0_4007_d2c7c1f2954f)) 
