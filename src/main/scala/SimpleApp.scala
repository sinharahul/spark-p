import org.apache.spark.sql.SparkSession

object SimpleApp {
  def main(args: Array[String]) {
    val logFile = "/Users/rahulsinha/projects/guestbook/project.clj" // Should be some file on your system
    val spark = SparkSession.builder.master("local[*]").appName("Simple Application").getOrCreate()
    val logData = spark.read.textFile(logFile).cache()
    val numAs = logData.filter(line => line.contains("project")).count()
    val numBs = logData.filter(line => line.contains("b")).count()
    println(s"Lines with project: $numAs, Lines with b: $numBs")
    spark.stop()
  }
}