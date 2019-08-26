package scalamachinelearning.ch1

import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler}
import org.apache.spark.sql.SparkSession


object Preprocess {
  def main(args: Array[String]) {
    val train = "/Users/rahulsinha/Downloads/allstate-claims-severity/train.csv" // Should be some file on your system
    val test = "/Users/rahulsinha/Downloads/allstate-claims-severity/test.csv" // Should be some file on your system
    var trainSample = 1.0
    var testSample = 1.0
    val spark = SparkSession.builder.master("local[*]").appName("Simple Application").getOrCreate()
    val logData = spark.read.textFile(train).cache()
    import spark.implicits._
    println("Reading data from " + train + " file")
    val trainInput = spark.read
                     .option("header", "true")
                     .option("inferSchema", "true")
                     .format("com.databricks.spark.csv")
                     .load(train)
                     .cache
    val testInput = spark.read
                    .option("header", "true")
                    .option("inferSchema", "true")
                    .format("com.databricks.spark.csv")
                    .load(test)
                    .cache
    println(trainInput.printSchema())
    println(trainInput.count())
    println("Preparing data for training model")
    var data = trainInput.withColumnRenamed("loss", "label")
               .sample(false, trainSample)
    var DF = data.na.drop()
    if (data == DF)
      println("No null values in the DataFrame")
    else{   println("Null values exist in the DataFrame")
           data = DF
        }
    val seed = 12345L
    val splits = data.randomSplit(Array(0.75, 0.25), seed)
    val (trainingData, validationData) = (splits(0), splits(1))
    trainingData.cache
    validationData.cache
    val testData = testInput.sample(false, testSample).cache
    def isCateg(c: String): Boolean = c.startsWith("cat")
    def categNewCol(c: String): String = if (isCateg(c)) s"idx_${c}" else c
    def removeTooManyCategs(c: String): Boolean = !(c matches "cat(109$|110$|112$|113$|116$)")
    def onlyFeatureCols(c: String): Boolean = !(c matches "id|label")
    val featureCols = trainingData.columns
                     .filter(removeTooManyCategs)
                     .filter(onlyFeatureCols)
                     .map(categNewCol)
    val stringIndexerStages = trainingData.columns.filter(isCateg)
                              .map(c => new StringIndexer()
                                .setInputCol(c)

                                .setOutputCol(categNewCol(c))
                                .fit(trainInput.select(c).union(testInput.select(c))))
    val assembler = new VectorAssembler()
                        .setInputCols(featureCols)
                        .setOutputCol("features")
    spark.stop()
  }
}