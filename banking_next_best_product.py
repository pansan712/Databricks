from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer
import mlflow
from delta.tables import *

# Create SparkSession
spark = SparkSession.builder.appName("NextBestProductRecommendation").getOrCreate()

# 1. Data Ingestion & Preparation (Delta Lake)
delta_table_path = "/path/to/banking_transactions"

# Load data from Delta Lake
df = spark.read.format("delta").load(delta_table_path)

# Prepare data for collaborative filtering
# Assume df has columns: customer_id, product_id, rating (e.g., transaction amount, engagement)
# Filter relevant transactions and aggregate if necessary

# 2. Collaborative Filtering (ALS)

# Data Split
(training, test) = df.randomSplit([0.8, 0.2])

# StringIndexer for categorical features
indexer_customer = StringIndexer(inputCol="customer_id", outputCol="customer_id_indexed")
indexer_product = StringIndexer(inputCol="product_id", outputCol="product_id_indexed")

# ALS Model
als = ALS(userCol="customer_id_indexed", itemCol="product_id_indexed", ratingCol="rating",
          coldStartStrategy="drop", implicitPrefs=True)  # Implicit feedback
pipeline = Pipeline(stages=[indexer_customer, indexer_product, als])

# Hyperparameter Tuning (Optional)
paramGrid = ParamGridBuilder() \
    .addGrid(als.rank, [8, 12]) \
    .addGrid(als.regParam, [0.01, 0.1]) \
    .build()

crossval = CrossValidator(estimator=pipeline,
                          estimatorParamMaps=paramGrid,
                          evaluator=RegressionEvaluator(metricName="rmse"),
                          numFolds=3)

# 3. MLflow Tracking & Model Evaluation
mlflow.set_experiment("/next_best_product")
with mlflow.start_run():
    # Train with Cross Validation (Optional)
    # cvModel = crossval.fit(training)
    # model = cvModel.bestModel

    # Train model directly (without hyperparameter tuning)
    model = pipeline.fit(training)
    
    # Make predictions on the test set
    predictions = model.transform(test)

    # Evaluate model
    evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")
    rmse = evaluator.evaluate(predictions)
    print(f"Root-mean-square error = {rmse:.4f}")
    
    # Log model and parameters
    mlflow.spark.log_model(model, "als_model")
    # mlflow.log_params(cvModel.bestModel.stages[-1].extractParamMap()) # If using CrossValidator
    mlflow.log_metric("rmse", rmse)


# 4. Generating Recommendations
# Get top-N product recommendations for all users
userRecs = model.recommendForAllUsers(10)

# Transform back to original product IDs
userRecs = userRecs.withColumn("product_id", model.stages[1].labels[userRecs.recommendations.product_id_indexed])


# Display Results
display(userRecs)
