package explore

import breeze.numerics.round
import org.apache.spark.ml.feature.{StandardScaler, VectorAssembler}
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession
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
    //val arr=MLUtils.kFold(amountDf.toDF().toJavaRDD.rdd,5,0)
   // amountDf.stat.sampleBy()
    /**
    for(a<-arr){
      val (x,y) = a
      println(s"x=${x.foreach(println)} y= ${y.foreach(println)}")
    }
      */
    //foldedRDD.foreach(rdd=>rdd.toDF().show(1))
    //  println("No Frauds", round(df['Class'].value_counts()[0]/len(df) * 100,2), '% of the dataset')
    //println('Frauds', round(df['Class'].value_counts()[1]/len(df) * 100,2), '% of the dataset')
   // val amount = new StandardScaler().fit(trainInput.columns.filter(c=>c.equals("Amount"))
//    val amountDf=new StandardScaler().setInputCol("Amount").setOutputCol("scaledAmount")
//        .fit(trainInput).transform(trainInput)
    /**
    import org.apache.spark.sql.functions._
    import spark.implicits._
    //https://stackoverflow.com/posts/50166298/edit
    val (mean_amount, std_amount) = trainInput.select(mean("Amount"), stddev("Amount"))
      .as[(Double, Double)]
      .first()
  //  amountDf.show(10)
    val trainInput1=trainInput.withColumn("amount_scaled", ($"Amount" - mean_amount) / std_amount)
    trainInput1.show(10)
  */
    spark.stop()
  }

}