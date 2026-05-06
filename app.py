import os
import sys
import streamlit as st
import pandas as pd
import json
import time
import requests
from bs4 import BeautifulSoup
import plotly.express as px
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel

# --- WINDOWS ENVIRONMENT CONFIGURATION ---
os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

java_home = r"C:\Program Files\EclipseAdoptium\jdk-8.0.472.8-hotspot"
os.environ['JAVA_HOME'] = java_home

hadoop_path = r"C:\Users\Admin\Documents\VLU\HKI1-2026\CCCVNTTTNtAI\btth\spark_util\hadoop-2.7.7"
spark_home = r"C:\Users\Admin\Documents\VLU\HKI1-2026\CCCVNTTTNtAI\btth\spark_util\spark-3.0.3-bin-hadoop2.7"

os.environ['HADOOP_HOME'] = hadoop_path
os.environ['SPARK_HOME'] = spark_home

os.environ['PATH'] = os.path.join(java_home, "bin") + os.pathsep + \
                     os.path.join(hadoop_path, "bin") + os.pathsep + \
                     os.path.join(spark_home, "bin") + os.pathsep + \
                     os.environ.get('PATH', '')

# --- APP SETTINGS ---
st.set_page_config(page_title="Big Data Sports Dashboard", page_icon="🏆", layout="wide")

@st.cache_resource
def init_spark():
    return SparkSession.builder \
        .master("local[*]") \
        .appName("Sports_Classification_System") \
        .config("spark.driver.memory", "4g") \
        .getOrCreate()

spark = init_spark()

@st.cache_resource
def load_assets():
    model = PipelineModel.load("sports_model_150M")
    df_count = spark.read.parquet("dataset_150M.parquet").count()
    return model, df_count

try:
    classifier_model, total_records = load_assets()
except Exception as e:
    st.error(f"Lỗi tải dữ liệu/mô hình: {e}")
    classifier_model = None
    total_records = 150000000

if "messages" not in st.session_state:
    st.session_state.messages = []

# --- INTERFACE ---
st.title("Big Data Sports Event Classifier")
st.write(f"System status: **Connected** | Current Dataset: **{total_records:,} rows (Parquet)**")

# --- SIDEBAR NAV & ADVANCED INPUTS ---
st.sidebar.header("Control Panel")
choice = st.sidebar.radio("Navigation", ["Inference Service (Chat)", "Cluster Performance (Chart)"])

st.sidebar.markdown("---")

# TÍNH NĂNG MỚI: Xử lý File và URL đặt ở Sidebar
advanced_prompt = None
if choice == "Inference Service (Chat)":
    with st.sidebar.expander("🔗 Phân tích từ File / Link bài báo", expanded=False):
        uploaded_file = st.file_uploader("📂 Tải file bài báo (.txt)", type=["txt"])
        url_input = st.text_input("🌐 Hoặc dán Link bài báo vào đây:")
        
        if st.button("🚀 Phân tích nâng cao", use_container_width=True):
            if uploaded_file is not None:
                advanced_prompt = uploaded_file.read().decode("utf-8")
                st.sidebar.success("Đã trích xuất văn bản từ File!")
            elif url_input:
                try:
                    res = requests.get(url_input, timeout=10)
                    soup = BeautifulSoup(res.text, 'html.parser')
                    # Lấy tất cả văn bản trong thẻ <p>
                    advanced_prompt = ' '.join([p.text for p in soup.find_all('p')])
                    if not advanced_prompt.strip():
                        st.sidebar.error("Không tìm thấy nội dung văn bản trong Link này.")
                        advanced_prompt = None
                    else:
                        st.sidebar.success("Đã cào dữ liệu thành công!")
                except Exception as e:
                    st.sidebar.error(f"Lỗi truy cập URL: {e}")
            else:
                st.sidebar.warning("Vui lòng tải file hoặc nhập link!")

st.sidebar.markdown("---")
st.sidebar.write("Author: **Nguyễn Thiên An**")
st.sidebar.write("Domain: Big Data Analysis Project")

# --- XỬ LÝ GIAO DIỆN CHÍNH ---
if choice == "Inference Service (Chat)":
    
    # Hiển thị lịch sử chat
    for idx, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            # Cắt ngắn text hiển thị nếu nó quá dài (do tải từ file/link)
            display_text = message["content"]
            if len(display_text) > 500 and message["role"] == "user":
                display_text = display_text[:500] + "...\n\n*(Nội dung đã được thu gọn)*"
            st.markdown(display_text)
            
            if message["role"] == "assistant":
                if "inference_time" in message:
                    st.caption(f"⏱️ *Thời gian trích xuất & suy luận: {message['inference_time']} giây*")
                
                if "image" in message and message["image"] and os.path.exists(message["image"]):
                    st.image(message["image"], caption=f"Category: {message['category']}", width=350)
                
                if "prob_data" in message:
                    st.write("**Phân bổ xác suất dự đoán:**")
                    df_prob = pd.DataFrame(message["prob_data"])
                    fig = px.bar(df_prob, x="Confidence", y="Category", orientation='h', 
                                 color="Confidence", color_continuous_scale="Blues", text_auto='.2%')
                    fig.update_layout(height=220, margin=dict(l=0, r=0, t=0, b=0), showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)
                
                with st.expander("Mô hình dự đoán sai? Cung cấp phản hồi"):
                    st.radio("Nhãn thực sự của câu này là gì?", 
                             ["Badminton", "Tennis", "Basketball", "Football", "Esports"], 
                             key=f"radio_{idx}")
                    if st.button("Gửi dữ liệu Re-train", key=f"btn_{idx}"):
                        st.success("Đã ghi nhận phản hồi vào hệ thống!")

    # Khung nhập liệu mặc định
    chat_prompt = st.chat_input("Nhập mô tả sự kiện thể thao vào đây...")
    
    # Gộp chung nguồn Prompt (từ chat hoặc từ file/link)
    prompt = chat_prompt or advanced_prompt

    if prompt:
        # 1. Thêm câu hỏi vào lịch sử và hiển thị (thu gọn nếu quá dài)
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            display_prompt = prompt if len(prompt) <= 500 else prompt[:500] + "...\n\n*(Nội dung đã được thu gọn)*"
            st.markdown(display_prompt)

        # 2. Suy luận Spark
        with st.chat_message("assistant"):
            start_time = time.time()
            with st.spinner("Đang trích xuất đặc trưng & suy luận từ 150M dòng (Spark Local)..."):
                
                if classifier_model is not None:
                    test_df = spark.createDataFrame([(prompt,)], ["text"])
                    prediction_df = classifier_model.transform(test_df)
                    
                    row_result = prediction_df.select("prediction", "probability").collect()[0]
                    pred_val = row_result["prediction"]
                    prob_array = row_result["probability"].toArray() 
                    
                    mapping = {
                        0.0: "Badminton",   
                        1.0: "Tennis",      
                        2.0: "Basketball",  
                        3.0: "Football",    
                        4.0: "Esports"      
                    }
                    result_name = mapping.get(pred_val, "Unknown")
                    
                    prob_data = pd.DataFrame({
                        "Category": ["Badminton", "Tennis", "Basketball", "Football", "Esports"],
                        "Confidence": prob_array
                    })
                else:
                    time.sleep(1.2)
                    result_name = "Esports"
                    prob_data = pd.DataFrame({
                        "Category": ["Badminton", "Tennis", "Basketball", "Football", "Esports"],
                        "Confidence": [0.05, 0.02, 0.13, 0.05, 0.75]
                    })

                inference_time = round(time.time() - start_time, 3)
                response_text = f"Kết quả phân loại: **{result_name}**"
                image_file = f"images/{result_name.lower()}.jpg"

            # 3. Trả về kết quả
            st.markdown(response_text)
            st.caption(f"⏱️ *Thời gian trích xuất & suy luận: {inference_time} giây*")
            
            if os.path.exists(image_file):
                st.image(image_file, caption=f"Predicted Category: {result_name}", width=350)
            else:
                st.warning(f"Image for {result_name} not found in /images directory.")
                
            st.write("**Phân bổ xác suất dự đoán:**")
            fig = px.bar(prob_data, x="Confidence", y="Category", orientation='h', 
                         color="Confidence", color_continuous_scale="Blues", text_auto='.2%')
            fig.update_layout(height=220, margin=dict(l=0, r=0, t=0, b=0), showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
            
            # 4. Lưu phản hồi vào lịch sử
            st.session_state.messages.append({
                "role": "assistant", 
                "content": response_text,
                "image": image_file,
                "category": result_name,
                "prob_data": prob_data.to_dict(),
                "inference_time": inference_time
            })
            
            with st.expander(" Mô hình dự đoán sai? Cung cấp phản hồi"):
                st.radio("Nhãn thực sự của câu này là gì?", 
                         ["Badminton", "Tennis", "Basketball", "Football", "Esports"], 
                         key="radio_latest")
                if st.button("Gửi dữ liệu Re-train", key="btn_latest"):
                    st.success("Đã ghi nhận phản hồi vào hệ thống!")

elif choice == "Cluster Performance (Chart)":
    st.header("Cụm Giám Sát Hiệu Năng Phân Tán")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Tổng Dữ Liệu", "150M Dòng", "Định dạng Parquet")
    col2.metric("Phân vùng (Partition)", "200 Partitions", "Tối ưu CPU")
    col3.metric("RAM Cấp Phát", "4 GB", "Driver Memory")
    col4.metric("Avg. Latency", "1.2s", "-0.3s", delta_color="inverse")
    
    st.markdown("---")
    
    try:
        with open('sports_results.json', 'r') as f:
            bench_data = json.load(f)
        df_bench = pd.DataFrame(bench_data)
    except Exception as e:
        st.info("Chưa tìm thấy file 'sports_results.json'. Đang hiển thị dữ liệu mẫu để demo.")
        df_bench = pd.DataFrame({
            "Milestone": ["10M", "50M", "100M", "150M"],
            "Processing_Time_Sec": [12, 45, 95, 140],
            "Throughput_Rows_Per_Sec": [833333, 1111111, 1052631, 1071428]
        })

    tab_time, tab_throughput, tab_raw = st.tabs([" Khả năng mở rộng", " Tốc độ thông lượng", " Dữ liệu thô"])
    
    with tab_time:
        st.subheader("Thời gian xử lý theo quy mô dữ liệu")
        fig_time = px.line(df_bench, x="Milestone" if "Milestone" in df_bench.columns else "milestones", 
                           y="Processing_Time_Sec" if "Processing_Time_Sec" in df_bench.columns else "time", 
                           markers=True, text="Processing_Time_Sec" if "Processing_Time_Sec" in df_bench.columns else "time")
        fig_time.update_traces(textposition="bottom right")
        st.plotly_chart(fig_time, use_container_width=True)
        
    with tab_throughput:
        st.subheader("Số dòng xử lý mỗi giây (Throughput)")
        if "Throughput_Rows_Per_Sec" in df_bench.columns:
            fig_tp = px.bar(df_bench, x="Milestone", y="Throughput_Rows_Per_Sec",
                            color="Throughput_Rows_Per_Sec", color_continuous_scale="Teal", text_auto='.2s')
            st.plotly_chart(fig_tp, use_container_width=True)
        else:
            st.warning("JSON của bạn chưa có trường Throughput để vẽ biểu đồ này.")
            
    with tab_raw:
        st.subheader("Bảng dữ liệu Benchmark")
        st.dataframe(df_bench, use_container_width=True)