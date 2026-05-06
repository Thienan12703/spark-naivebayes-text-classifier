import time
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, rand, when, concat_ws, element_at, array, lit

def create_synthetic_data_pipeline():
    """
    Module sinh dữ liệu mô phỏng (Synthetic Data Generation) quy mô lớn.
    Sử dụng kỹ thuật hoán vị tổ hợp các Seed Keywords từ BBC Sports, ESPN.
    Output: Định dạng Columnar Parquet siêu nén.
    """
    spark = SparkSession.builder \
        .master("local[*]") \
        .appName("Synthetic_Data_Generator_150M") \
        .config("spark.sql.shuffle.partitions", "200") \
        .config("spark.memory.fraction", "0.8") \
        .getOrCreate()

    print("Khởi động PySpark Engine: Chuẩn bị sinh 150 triệu bản ghi...")
    start_time = time.time()

    subjects = array([lit("The player"), lit("The champion"), lit("The captain"), lit("The rookie")])
    verbs = array([lit("executed"), lit("performed"), lit("managed"), lit("delivered")])

    football_terms = array([lit("a long pass past the goalkeeper"), lit("a beautiful goal into the net"), lit("a precise cross into the penalty box")])
    badminton_terms = array([lit("a powerful jump smash"), lit("a delicate drop shot near the net"), lit("a precise baseline clear")])
    basketball_terms = array([lit("a spectacular slam dunk"), lit("a contested jumper right at the buzzer"), lit("a high pick-and-roll")])
    tennis_terms = array([lit("an unstoppable ace at 120 mph"), lit("a crisp backhand volley"), lit("heavy topspin groundstrokes")])
    esports_terms = array([lit("a perfect gank to secure the nexus"), lit("a flawless teleport flank"), lit("deep vision control around the Baron pit")])

    total_records = 150000000
    df = spark.range(0, total_records)


    df_labeled = df.withColumn("label", (rand() * 5).cast("int").cast("double"))

  
    df_synthetic = df_labeled.withColumn(
        "text",
        when(col("label") == 0.0, concat_ws(" ", element_at(subjects, (rand()*4 + 1).cast("int")), element_at(verbs, (rand()*4 + 1).cast("int")), element_at(football_terms, (rand()*3 + 1).cast("int"))))
        .when(col("label") == 1.0, concat_ws(" ", element_at(subjects, (rand()*4 + 1).cast("int")), element_at(verbs, (rand()*4 + 1).cast("int")), element_at(badminton_terms, (rand()*3 + 1).cast("int"))))
        .when(col("label") == 2.0, concat_ws(" ", element_at(subjects, (rand()*4 + 1).cast("int")), element_at(verbs, (rand()*4 + 1).cast("int")), element_at(basketball_terms, (rand()*3 + 1).cast("int"))))
        .when(col("label") == 3.0, concat_ws(" ", element_at(subjects, (rand()*4 + 1).cast("int")), element_at(verbs, (rand()*4 + 1).cast("int")), element_at(tennis_terms, (rand()*3 + 1).cast("int"))))
        .otherwise(concat_ws(" ", element_at(subjects, (rand()*4 + 1).cast("int")), element_at(verbs, (rand()*4 + 1).cast("int")), element_at(esports_terms, (rand()*3 + 1).cast("int"))))
    )

    output_path = "dataset_150M.parquet"
    print(f"Đang ghi dữ liệu phân tán ra định dạng Parquet tại: {output_path}")
    
    df_synthetic.select("text", "label").write \
        .mode("overwrite") \
        .parquet(output_path)

    end_time = time.time()
    print(f"Hoàn tất sinh {total_records:,} dòng dữ liệu!")
    print(f"Tổng thời gian thực thi: {(end_time - start_time) / 60:.2f} phút.")

if __name__ == "__main__":
    create_synthetic_data_pipeline()








    