package scalamachinelearning.ch1

import org.apache.spark.sql.SparkSession

object InsuranceClaims {
  def main(args: Array[String]) {
    val insuranceFile = "/Users/rahulsinha/Downloads/allstate-claims-severity/train.csv" // Should be some file on your system

    val spark = SparkSession.builder.master("local[*]").appName("Simple Application").getOrCreate()
    val logData = spark.read.textFile(insuranceFile).cache()
    val trainInput = spark.read
                     .option("header", "true")
                     .option("inferSchema", "true")
                     .format("com.databricks.spark.csv")
                     .load(insuranceFile)
                     .cache
    trainInput.show(1)
    println(trainInput.printSchema())
    println(trainInput.count())
    trainInput.select("id", "cat1", "cat2", "cat3", "cont1", "cont2", "cont3", "loss").show()
    val newDF = trainInput.withColumnRenamed("loss", "label")
    newDF.createOrReplaceTempView("insurance")
    spark.sql("SELECT avg(insurance.label) as AVG_LOSS FROM insurance").show()
    spark.stop()
  }
}