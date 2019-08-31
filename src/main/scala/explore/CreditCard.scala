package explore

import breeze.numerics.round
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.feature.{StandardScaler, VectorAssembler}
import org.apache.spark.ml.stat.Correlation
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, Dataset, Row, SparkSession}
//https://www.kaggle.com/mlg-ulb/creditcardfraud#creditcard.csv
//https://www.kaggle.com/janiobachmann/credit-fraud-dealing-with-imbalanced-datasets
/**
  * Detect credit card fraud
  */
object CreditCard {

  def main(args: Array[String]) {
    val creditCard = "/Users/rahulsinha/Downloads/creditcard.csv" // Should be some file on your system
    val spark = SparkSession.builder.master("local[*]").appName("Credit Card").getOrCreate()
    val logData = spark.read.textFile(creditCard).cache()
    val trainInput = spark.read
                     .option("header", "true")
                     .option("inferSchema", "true")
                     .format("com.databricks.spark.csv")
                     .load(creditCard)
                     .cache
    var df = trainInput.na.drop()
    if (trainInput == df)
      println("No null values in the DataFrame")
    else{   println("Null values exist in the DataFrame")
    }
    println(trainInput.printSchema())
    println(trainInput.count())
    trainInput.select("Class").show()
    val classDf=trainInput.select("Class")
    val noFrauds=classDf.filter(s=>s.get(0)==0)
    val frauds = classDf.filter(s=>s.get(0)==1)
    val nfCnt =(noFrauds.count().toFloat/classDf.count())*100.0
    val frdCnt:Double =(frauds.count().toFloat/classDf.count())*100
    println(s"Classdf = ${classDf.count()}")
    println(s"No fraud count ${noFrauds.count()}")
    println(s"fraud count ${frauds.count()}")
    println(s"No frauds are ${nfCnt} % of dataset .Frauds are ${frdCnt}" )
    val vectorAssemberAmount = new VectorAssembler().setInputCols(Array("Amount")).setOutputCol("amount_assem")
    val vectorAssemberTime = new VectorAssembler().setInputCols(Array("Time")).setOutputCol("time_assem")
    var newDF=vectorAssemberAmount.transform(trainInput)
    newDF=vectorAssemberTime.transform(newDF)
    var amountDf=new StandardScaler().setInputCol("amount_assem").setOutputCol("scaledAmount")
    .fit(newDF).transform(newDF)
    amountDf=new StandardScaler().setInputCol("time_assem").setOutputCol("scaledTime")
      .fit(amountDf).transform(amountDf)
    amountDf.show(10)
    amountDf=amountDf.drop("Amount","Time")
    amountDf.createOrReplaceTempView("creditcard")
    spark.sql(
      """
        SELECT count(*),Class from creditcard
         group by Class
      """.stripMargin).show(10)
    //0.0017304750013189597
    val fractions = Map(1 -> 1.0 , 0 -> 0.0017304750013189597)
    val churnDF = amountDf.stat.sampleBy("Class", fractions, 12345L)
    churnDF.groupBy("Class").count.show()

    def createRandomDF(amountDf:DataFrame)={
      val sampledAmountDf=amountDf.sample(1)
      val fraud_df:Dataset[Row]=sampledAmountDf.where("Class=1")
      val non_fraud_df:Dataset[Row]=sampledAmountDf.where("Class=0").limit(492)
      fraud_df.union(non_fraud_df)
    }

    val new_df=createRandomDF(amountDf)
    new_df.createOrReplaceTempView("creditcard1")
    spark.sql(
      """
        SELECT count(*),Class from creditcard1
         group by Class
      """.stripMargin).show(10)
    new_df.show(10)
    def selectCols(newDf:DataFrame):Array[String]={
      newDf.columns.filter(col=>col.startsWith("V")||col.startsWith("scaled"))
    }
    selectCols(new_df).foreach(println)
    val va=new VectorAssembler().setInputCols(selectCols(new_df))
      .setOutputCol("features")
    val numFolds = 10
    val MaxIter: Seq[Int] = Seq(100)
    val RegParam: Seq[Double] = Seq(1.0) // L2 regularization param, set 1.0 with L1 regularization
    val Tol: Seq[Double] = Seq(1e-8)// for convergence tolerance for iterative algorithms
    val ElasticNetParam: Seq[Double] = Seq(0.0001) //Combination of L1 & L2
    val lr = new LogisticRegression()
             .setLabelCol("Class")
             .setFeaturesCol("features")

    import org.apache.spark._
    import org.apache.spark.sql.SparkSession
    import org.apache.spark.sql.functions._
    import org.apache.spark.ml.classification.{BinaryLogisticRegressionSummary, LogisticRegression, LogisticRegressionModel}
    import org.apache.spark.ml.Pipeline
    import org.apache.spark.ml.tuning.{ParamGridBuilder, CrossValidator}
    import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
    import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
    import org.apache.spark.mllib.linalg._
    import org.apache.spark.mllib.stat.Statistics
    import org.apache.spark.rdd.RDD
    val pipeline = new Pipeline()
                    .setStages(Array( va, lr))

    val paramGrid = new ParamGridBuilder()
                        .addGrid(lr.maxIter, MaxIter)
                        .addGrid(lr.regParam, RegParam)
                        .addGrid(lr.tol, Tol)
                        .addGrid(lr.elasticNetParam, ElasticNetParam)
                        .build()
    val evaluator = new BinaryClassificationEvaluator()
                        .setLabelCol("Class")
                        .setRawPredictionCol("prediction")
    val crossval = new CrossValidator()
                      .setEstimator(pipeline)
                      .setEvaluator(evaluator)
                      .setEstimatorParamMaps(paramGrid)
                      .setNumFolds(numFolds)
    val arr=new_df.randomSplit(Array(0.8,0.2))
    val (trainDF,testDF)=(arr(0),arr(1))
    trainDF.cache()
    val cvModel = crossval.fit(trainDF)
    val predictions = cvModel.transform(testDF)
    val result = predictions.select("Class", "prediction", "probability")
    val resutDF = result.withColumnRenamed("prediction", "Predicted_label")
    resutDF.show(10)
    val accuracy = evaluator.evaluate(predictions)
    println("Classification accuracy: " + accuracy)
    spark.stop()
  }

}