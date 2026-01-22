from pyspark.sql import SparkSession
from pyspark.sql import SparkSession
import pandas as pd

def handle_youtube_query(sql_query: str) -> pd.DataFrame:
    spark = SparkSession.builder \
        .appName("MultiAgentNL2SQL") \
        .enableHiveSupport() \
        .getOrCreate()

    try:
        spark_df = spark.sql(sql_query)
        pdf = spark_df.toPandas()
        return pdf
    except Exception as e:
        return pd.DataFrame({"error": [str(e)]})  # So that .empty will still work


if __name__ == '__main__':
    query = "describe youtube_data1.trending_videos;"
    print(handle_youtube_query(query))
