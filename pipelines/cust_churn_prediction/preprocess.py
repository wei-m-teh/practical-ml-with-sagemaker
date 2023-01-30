import argparse
import os
from pyspark.sql.types import IntegerType, BooleanType, DateType, StringType
from pyspark.sql.functions import col, sum, avg, count, explode, array, lit
from pyspark.sql import functions as F
from pyspark.sql import types as T
from pyspark.sql.functions import to_date
from pyspark.sql.functions import quarter
from pyspark.sql.functions import log
from pyspark.sql.functions import mean
from pyspark.sql.functions import when
from pyspark.sql.functions import datediff
import dateutil.parser
from pyspark.sql.functions import log1p
from pyspark.ml.feature import Imputer
from pyspark.ml.feature import StringIndexer
import pyspark
from pyspark.sql import SparkSession

def find_mode(x):
  """Find the value with highest occurence in the list"""
  return max(set(x), key=x.count)

def main():
    parser = argparse.ArgumentParser(description="app inputs and outputs")
    parser.add_argument("--s3_input_bucket", type=str, help="s3 input bucket")
    parser.add_argument("--s3_input_key_prefix", type=str, help="s3 input key prefix")
    parser.add_argument("--s3_output_bucket", type=str, help="s3 output bucket")
    parser.add_argument("--s3_output_key_prefix", type=str, help="s3 output key prefix")
    args = parser.parse_args()
    
    spark = SparkSession.builder.config("mapreduce.fileoutputcommitter.marksuccessfuljobs", "false").config("spark.driver.maxResultSize", "30g").config("parquet.enable.summary-metadata", "false").getOrCreate()
    
    train_df = spark.read.format('csv').options(header='true', inferSchema='true').load(f's3://{args.s3_input_bucket}/{args.s3_input_key_prefix}/train.csv')
    trans_df = spark.read.format('csv').options(header='true', inferSchema='true').load(f's3://{args.s3_input_bucket}/{args.s3_input_key_prefix}/transactions.csv')
    trans_df = trans_df.withColumn("transaction_date",trans_df.transaction_date.cast(StringType())).\
    withColumn("membership_expire_date",trans_df.membership_expire_date.cast(StringType()))

    find_mode_udf = F.udf(find_mode, T.IntegerType())
    payment_plan_days_df = trans_df.groupby("msno").agg(find_mode_udf(F.collect_list('payment_plan_days')).alias("max_plan_days"))

    payment_method_id_df = trans_df.groupby("msno").agg(find_mode_udf(F.collect_list('payment_method_id')).alias("max_payment_method_id"))

    revenue = trans_df.groupby("msno").agg(sum("actual_amount_paid").alias("total_amount_paid"))

    auto_renew_df = trans_df.groupby("msno").agg(find_mode_udf(F.collect_list('is_auto_renew')).alias("max_auto_renew"))

    is_canceled_df = trans_df.groupby("msno").agg(sum("is_cancel").alias("total_is_canceled"))

    cust_payment_method_df = trans_df.groupby("msno").agg(count("payment_method_id").alias("payment_method_id"))

    trans_df = trans_df.withColumn("quarter",quarter(to_date(trans_df.transaction_date, "yyyyMMdd"))). \
            withColumn("quarter_end",quarter(to_date(trans_df.membership_expire_date, "yyyyMMdd")))

    quarter_df = trans_df.groupby("msno").agg(find_mode_udf(F.collect_list('quarter')).alias("max_quarter"))
    merged_df = cust_payment_method_df.join(payment_method_id_df, on="msno").join(payment_plan_days_df, on="msno").join(revenue, on="msno").join(auto_renew_df, on="msno").join(is_canceled_df, on="msno").join(quarter_df, on="msno")

    merged_df = merged_df.withColumnRenamed("payment_method_id", "regist_trans")\
           .withColumnRenamed("max_payment_method_id", "mst_frq_pay_met")\
           .withColumnRenamed("max_plan_days", "mst_frq_plan_days")\
           .withColumnRenamed("total_amount_paid", "revenue")\
           .withColumnRenamed("max_auto_renew", "is_auto_renew")\
           .withColumnRenamed("total_is_canceled", "regist_cancels")\
           .withColumnRenamed("max_quarter", "qtr_trans")

    members_df = spark.read.format('csv').options(header='true', inferSchema='true').load(f's3://{args.s3_input_bucket}/{args.s3_input_key_prefix}/members_v3.csv')

    members_df = members_df.na.fill("other")
    logs_df = spark.read.format('csv').options(header='true', inferSchema='true').load(f's3://{args.s3_input_bucket}/{args.s3_input_key_prefix}/user_logs.csv')
    logs_df = logs_df.withColumn("total_secs",log(logs_df.total_secs))
    logs_mean_df = logs_df.groupby("msno").agg(mean("num_25").alias("num_25"), mean("num_50").alias("num_50"), mean("num_75").alias("num_75"), mean("num_985").alias("num_985"), mean("num_100").alias("num_100"), mean("num_unq").alias("num_unq"), mean("total_secs").alias("total_secs"))

    all_df = train_df.join(merged_df, on="msno", how="left").join(members_df, on="msno", how="left").join(logs_mean_df, on="msno", how="left")
    upper_thresh = 80
    all_df = all_df.withColumn("bd", when(all_df.bd > upper_thresh, 0))
    all_df = all_df.na.fill(value=0,subset=["bd"])
    end_dt = dateutil.parser.parse('2017-03-31') 
    all_df = all_df.withColumn("tenure",datediff(to_date(lit(end_dt), "yyyy-MM-dd"), to_date(all_df.registration_init_time.cast(StringType()), "yyyyMMdd")) / 365).cache()
    all_df_cnt = all_df.count()
    filtered_cols_registered_via = all_df.groupBy('registered_via').agg(count("registered_via").alias("count_registered_via")).filter(col("count_registered_via") / all_df_cnt < 0.005).select("registered_via").rdd.map(lambda x : x[0]).collect()
    filtered_df = all_df.withColumn("registered_via", when(all_df.registered_via.isin(filtered_cols_registered_via) ,"other").\
                                    when(all_df.registered_via.isNull() ,"other").otherwise(all_df.registered_via))

    filtered_cols_mst_frq_pay_met = filtered_df.groupBy('mst_frq_pay_met').agg(count("mst_frq_pay_met").alias("count_mst_frq_pay_met")).filter(col("count_mst_frq_pay_met") / all_df_cnt < 0.005).select("mst_frq_pay_met").rdd.map(lambda x : x[0]).collect()

    filtered_df = filtered_df.withColumn("mst_frq_pay_met", when(filtered_df.mst_frq_pay_met.isin(filtered_cols_mst_frq_pay_met) ,"other").\
                                         when(filtered_df.mst_frq_pay_met.isNull() ,"other").otherwise(filtered_df.mst_frq_pay_met))

    filtered_cols_city = filtered_df.groupBy('city').agg(count("city").alias("count_city")).filter(col("count_city") / all_df_cnt < 0.005).select("city").rdd.map(lambda x : x[0]).collect()

    filtered_df = filtered_df.withColumn("city", when(filtered_df.city.isin(filtered_cols_city) ,"other").\
                                         when(filtered_df.city.isNull() ,"other").otherwise(filtered_df.city))

    filtered_df = filtered_df.withColumn("regist_trans", log1p("regist_trans")).\
    withColumn("mst_frq_plan_days", log1p("mst_frq_plan_days")).\
    withColumn("revenue", log1p("revenue")).\
    withColumn("regist_cancels", log1p("regist_cancels"))

    train, test, validation = filtered_df.randomSplit(weights=[0.7,0.2, 0.1], seed=200)

    final_train = train.drop('registration_init_time')
    final_test = test.drop('registration_init_time')
    final_val = validation.drop('registration_init_time')
    
    imputer = Imputer()
    imputer.setInputCols(['num_25','num_50','num_75','num_985','num_100','num_unq','total_secs', 'tenure'])
    imputer.setOutputCols(['num_25_o','num_50_o','num_75_o','num_985_o','num_100_o','num_unq_o','total_secs_o', 'tenure_o'])
    model = imputer.fit(final_train)
    model.setInputCols(['num_25','num_50','num_75','num_985','num_100','num_unq','total_secs', 'tenure'])

    imputed_train = model.transform(final_train).cache()
    imputed_test = model.transform(final_test).cache()
    imputed_val = model.transform(final_val).cache()
    
    temp_imputed_train = imputed_train.drop('num_25','num_50','num_75','num_985','num_100','num_unq','total_secs', 'tenure')
    temp_imputed_test = imputed_test.drop('num_25','num_50','num_75','num_985','num_100','num_unq','total_secs', 'tenure')
    temp_imputed_val = imputed_val.drop('num_25','num_50','num_75','num_985','num_100','num_unq','total_secs', 'tenure')

    temp_imputed_train = temp_imputed_train.withColumnRenamed("num_25_o","num_25")\
    .withColumnRenamed("num_50_o","num_50")\
    .withColumnRenamed("num_75_o","num_75")\
    .withColumnRenamed("num_985_o","num_985")\
    .withColumnRenamed("num_100_o","num_100")\
    .withColumnRenamed("num_unq_o","num_unq")\
    .withColumnRenamed("total_secs_o","total_secs")\
    .withColumnRenamed("tenure_o","tenure")

    temp_imputed_test = temp_imputed_test.withColumnRenamed("num_25_o","num_25")\
    .withColumnRenamed("num_50_o","num_50")\
    .withColumnRenamed("num_75_o","num_75")\
    .withColumnRenamed("num_985_o","num_985")\
    .withColumnRenamed("num_100_o","num_100")\
    .withColumnRenamed("num_unq_o","num_unq")\
    .withColumnRenamed("total_secs_o","total_secs")\
    .withColumnRenamed("tenure_o","tenure")
    
    temp_imputed_val = temp_imputed_val.withColumnRenamed("num_25_o","num_25")\
    .withColumnRenamed("num_50_o","num_50")\
    .withColumnRenamed("num_75_o","num_75")\
    .withColumnRenamed("num_985_o","num_985")\
    .withColumnRenamed("num_100_o","num_100")\
    .withColumnRenamed("num_unq_o","num_unq")\
    .withColumnRenamed("total_secs_o","total_secs")\
    .withColumnRenamed("tenure_o","tenure")

    indexer = StringIndexer(inputCol="city", outputCol="cityIndex")
    model = indexer.fit(temp_imputed_train)
    indexed_train = model.transform(temp_imputed_train).drop("city").withColumnRenamed("cityIndex","city")
    indexed_test = model.transform(temp_imputed_test).drop("city").withColumnRenamed("cityIndex","city")
    indexed_val = model.transform(temp_imputed_val).drop("city").withColumnRenamed("cityIndex","city")

    indexed_train = indexed_train.withColumn("gender", when(indexed_train.gender.isNull(), "other").otherwise(indexed_train.gender))
    indexed_test = indexed_test.withColumn("gender", when(indexed_test.gender.isNull(), "other").otherwise(indexed_test.gender))
    indexed_val = indexed_val.withColumn("gender", when(indexed_val.gender.isNull(), "other").otherwise(indexed_val.gender))

    indexer = StringIndexer(inputCol="gender", outputCol="genderIndex").setHandleInvalid("keep")
    model = indexer.fit(indexed_train)
    indexed_train = model.transform(indexed_train).drop("gender").withColumnRenamed("genderIndex","gender")
    indexed_test = model.transform(indexed_test).drop("gender").withColumnRenamed("genderIndex","gender")
    indexed_val = model.transform(indexed_val).drop("gender").withColumnRenamed("genderIndex","gender")

    indexed_train = indexed_train.withColumn("registered_via", when(indexed_train.registered_via.isNull(), "other").otherwise(indexed_train.registered_via))
    indexed_test = indexed_test.withColumn("registered_via", when(indexed_test.registered_via.isNull(), "other").otherwise(indexed_test.registered_via))
    indexed_val = indexed_val.withColumn("registered_via", when(indexed_val.registered_via.isNull(), "other").otherwise(indexed_val.registered_via))

    indexer = StringIndexer(inputCol="registered_via", outputCol="registered_viaIndex")
    model = indexer.fit(indexed_train)
    indexed_train = model.transform(indexed_train).drop("registered_via").withColumnRenamed("registered_viaIndex","registered_via")
    indexed_test = model.transform(indexed_test).drop("registered_via").withColumnRenamed("registered_viaIndex","registered_via")
    indexed_val = model.transform(indexed_val).drop("registered_via").withColumnRenamed("registered_viaIndex","registered_via")

    indexed_train = indexed_train.withColumn("qtr_trans", when(indexed_train.qtr_trans.isNull(), "other").otherwise(indexed_train.qtr_trans))
    indexed_test = indexed_test.withColumn("qtr_trans", when(indexed_test.qtr_trans.isNull(), "other").otherwise(indexed_test.qtr_trans))
    indexed_val = indexed_val.withColumn("qtr_trans", when(indexed_val.qtr_trans.isNull(), "other").otherwise(indexed_val.qtr_trans))
    
    indexer = StringIndexer(inputCol="qtr_trans", outputCol="qtr_transIndex")
    model = indexer.fit(indexed_train)
    indexed_train = model.transform(indexed_train).drop("qtr_trans").withColumnRenamed("qtr_transIndex","qtr_trans")
    indexed_test = model.transform(indexed_test).drop("qtr_trans").withColumnRenamed("qtr_transIndex","qtr_trans")
    indexed_val = model.transform(indexed_val).drop("qtr_trans").withColumnRenamed("qtr_transIndex","qtr_trans")

    indexed_train = indexed_train.withColumn("mst_frq_pay_met", when(indexed_train.mst_frq_pay_met.isNull(), "other").otherwise(indexed_train.mst_frq_pay_met))
    indexed_test = indexed_test.withColumn("mst_frq_pay_met", when(indexed_test.mst_frq_pay_met.isNull(), "other").otherwise(indexed_test.mst_frq_pay_met))
    indexed_val = indexed_val.withColumn("mst_frq_pay_met", when(indexed_val.mst_frq_pay_met.isNull(), "other").otherwise(indexed_val.mst_frq_pay_met))

    indexer = StringIndexer(inputCol="mst_frq_pay_met", outputCol="mst_frq_pay_metIndex")
    model = indexer.fit(indexed_train)
    indexed_train = model.transform(indexed_train).drop("mst_frq_pay_met").withColumnRenamed("mst_frq_pay_metIndex","mst_frq_pay_met")
    indexed_test = model.transform(indexed_test).drop("mst_frq_pay_met").withColumnRenamed("mst_frq_pay_metIndex","mst_frq_pay_met")
    indexed_val = model.transform(indexed_val).drop("mst_frq_pay_met").withColumnRenamed("mst_frq_pay_metIndex","mst_frq_pay_met")

    indexed_train = indexed_train.withColumn("is_auto_renew", when(indexed_train.is_auto_renew.isNull(), "other").otherwise(indexed_train.is_auto_renew))
    indexed_test = indexed_test.withColumn("is_auto_renew", when(indexed_test.is_auto_renew.isNull(), "other").otherwise(indexed_test.is_auto_renew))
    indexed_val = indexed_val.withColumn("is_auto_renew", when(indexed_val.is_auto_renew.isNull(), "other").otherwise(indexed_val.is_auto_renew))
    
    indexer = StringIndexer(inputCol="is_auto_renew", outputCol="is_auto_renewIndex")
    model = indexer.fit(indexed_train)
    indexed_train = model.transform(indexed_train).drop("is_auto_renew").withColumnRenamed("is_auto_renewIndex","is_auto_renew")
    indexed_test = model.transform(indexed_test).drop("is_auto_renew").withColumnRenamed("is_auto_renewIndex","is_auto_renew")
    indexed_val = model.transform(indexed_val).drop("is_auto_renew").withColumnRenamed("is_auto_renewIndex","is_auto_renew")

    major_df = indexed_train.filter(col("is_churn") == 0)
    minor_df = indexed_train.filter(col("is_churn") == 1)
    ratio = int(major_df.count()/minor_df.count())
    print("ratio: {}".format(ratio))

    a = range(ratio)# duplicate the minority rows
    oversampled_df = minor_df.withColumn("dummy", explode(array([lit(x) for x in a]))).drop('dummy')

    final_train = major_df.unionAll(oversampled_df)

    final_train.coalesce(1).write.mode('overwrite').option("header",False).csv(f"s3://{args.s3_output_bucket}/{args.s3_output_key_prefix}/train")
    indexed_test.coalesce(1).write.mode('overwrite').option("header",False).csv(f"s3://{args.s3_output_bucket}/{args.s3_output_key_prefix}/test")
    indexed_val.coalesce(1).write.mode('overwrite').option("header",False).csv(f"s3://{args.s3_output_bucket}/{args.s3_output_key_prefix}/validation")
    
if __name__ == "__main__":
    main()