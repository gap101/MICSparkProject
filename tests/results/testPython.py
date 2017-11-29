from pyspark.sql import SparkSession
from pyspark.ml.evaluation import RegressionEvaluator

from pyspark.ml.regression import LinearRegression


spark = SparkSession.builder.appName("generatedApplicaton").getOrCreate() 
a94453aaa_6364_0361_e42e_1cf639a78f1b = spark.read.load('')
af1ed1f3b_8bb2_2382_d707_91362e48f4df = spark.read.load('')
LinearRegressor = LinearRegression(featuresCol = "features", labelCol = "label", predictionCol = "prediction", maxIter= 100, regParam= 0, elasticNetParam = 0 ,standardization = true, tol = 0.000001, fitIntercept = true, solver = "auto", weightCol = None, aggregationDepth = 2)
  
af56e50a7_5bdc_83df_61aa_fa65c3e4cb5a = LinearRegressor.fit(af1ed1f3b_8bb2_2382_d707_91362e48f4df)

evaluator = RegressionEvaluator(labelCol="", predictionCol="", metricName="rmse")
a606c5212_1532_1910_edb4_ed455da70cad = evaluator.evaluate(a94453aaa_6364_0361_e42e_1cf639a78f1b)
print("Summary : %s " % str(a606c5212_1532_1910_edb4_ed455da70cad)) 
