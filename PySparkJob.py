import io
import sys
from pyspark.sql import SparkSession
import pyspark.sql.functions as psf

def process(spark, input_file, target_path):
    #читаем файл с данными
    df = spark.read.option('header', 'true').parquet('clickstream.parquet')
    
    #создаем бинарные столбцы is_cpm и is_cpc
    df = df.withColumn('is_cpm', psf.when((psf.col('ad_cost_type') == 'CPM'), 1).otherwise(0))
    df = df.withColumn('is_cpc', psf.when((psf.col('ad_cost_type') == 'CPC'), 1).otherwise(0))
    
    #добавим в датафрейм столбцы с начальной и конечной датой показа объявления, числом кликов и просмотров
    df = df.groupBy('ad_id', 'ad_cost', 'has_video', 'target_audience_count', 'is_cpm', 'is_cpc').agg(
                                  psf.min('date').alias('date_start'), psf.max('date').alias('date_end'), 
                                  psf.count(psf.when(psf.col('event') == 'click', True)).alias('click_cnt'), 
                                  psf.count(psf.when(psf.col('event') == 'view', True)).alias('view_cnt'))
    
    #добавим столбцы с числом дней показа объявления и CTR
    df = df.withColumn('day_count', psf.datediff(df.date_end,df.date_start))
    df = df.withColumn('CTR', df.click_cnt/df.view_cnt)
    
    #удалим ненужные столбцы
    df = df.drop('date_start', 'date_end', 'click_cnt', 'view_cnt')
    
    #расставим столбцы датафрейма в правильном порядке, согласно требуемой структуры
    col_list = list(df)

    col_list[0], col_list[1], col_list[2], col_list[3], col_list[4], col_list[5], col_list[6], col_list[7] = \
    col_list[0], col_list[3], col_list[2], col_list[4], col_list[5], col_list[1], col_list[6], col_list[7]

    df = df[col_list]
    
    #разделим датафрейм на train и test в соотношении 75/25
    train, test = df.randomSplit([0.75, 0.25], 24)
    
    #создадим train и test файлы типа parquet 
    train.coalesce(1).write.option('header', 'true').parquet('result/train')
    test.coalesce(1).write.option('header', 'true').parquet('result/test')
    
def main(argv):
    input_path = argv[0]
    print("Input path to file: " + input_path)
    target_path = argv[1]
    print("Target path: " + target_path)
    spark = _spark_session()
    process(spark, input_path, target_path)


def _spark_session():
    return SparkSession.builder.appName('PySparkJob').getOrCreate()


if __name__ == "__main__":
    arg = sys.argv[1:]
    if len(arg) != 2:
        sys.exit("Input and Target path are require.")
    else:
        main(arg)
