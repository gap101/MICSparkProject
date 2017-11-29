from pyspark.sql import SparkSession
from pyspark.ml.evaluation import RegressionEvaluator

from pyspark.ml.regression import LinearRegression


spark = SparkSession.builder.appName("generatedApplicaton").getOrCreate() 
ab50b79ae_1526_fcae_e3b8_4000bb58db8b = spark.read.format("libsvm").load('D:\Spark\spark-2.2.0-bin-hadoop2.7\spark-2.2.0-bin-hadoop2.7\data\mllib\sample_linear_regression_data.txt')
(a5d65f8fc_3e47_2b84_4c25_61eb891a7523, a38f895a5_4fab_aa1e_1a03_adf9e1c58bee) = ab50b79ae_1526_fcae_e3b8_4000bb58db8b.randomSplit([0.7, 0.3])

LinearRegressor = LinearRegression(featuresCol = "features", labelCol = "label", predictionCol = "prediction", maxIter= 100, regParam= 0, elasticNetParam = 0 ,standardization = True, tol = 0.000001, fitIntercept = True, solver = "auto", aggregationDepth = 2)
  
a3d527322_9f5d_ce10_22f8_adbb232ce80f = LinearRegressor.fit(a5d65f8fc_3e47_2b84_4c25_61eb891a7523)

a6b64d920_d851_389c_0ee3_9ca0f5ef091d = a98c39b01_4c3c_3e4d_97bf_0bd5d5b3db3a.transform(a38f895a5_4fab_aa1e_1a03_adf9e1c58bee)
print("Summary : %s " % str(a3d527322_9f5d_ce10_22f8_adbb232ce80f)) 
evaluator = RegressionEvaluator(labelCol="", predictionCol="", metricName="rmse")
af59ea44a_9b0f_8a6b_dc90_994d14a2953c = evaluator.evaluate(a6b64d920_d851_389c_0ee3_9ca0f5ef091d)
print("Summary : %s " % str(af59ea44a_9b0f_8a6b_dc90_994d14a2953c)) 
