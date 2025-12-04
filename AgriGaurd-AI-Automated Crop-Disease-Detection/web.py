import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ------------------------
# Model Prediction Function
# ------------------------
def model_prediction(test_image):
    # Load the trained model (make sure the file is in the same folder)
    model = tf.keras.models.load_model('trained_model.keras')

    # Open the uploaded image and resize
    image = Image.open(test_image).resize((128, 128))
    input_arr = np.array(image)
    input_arr = np.expand_dims(input_arr, axis=0)  # Convert to batch
    prediction = model.predict(input_arr)
    result_index = np.argmax(prediction)
    return result_index

# ------------------------
# Custom CSS
# ------------------------
st.markdown("""
    <style>
    .main {background-color: #f5f5f5;}
    .stButton>button {
        color: white;
        background-color: #4CAF50;
        border-radius: 5px;
        padding: 0.5rem 1rem;
    }
    .stButton>button:hover {
        background-color: #45a049;
        color: white;
    }
    .stFileUploader>div>div>div>button {
        color: white;
        background-color: #2196F3;
    }
    .prediction-result {
        padding: 1rem;
        border-radius: 10px;
        background-color: #e8f5e9;
        margin: 1rem 0;
    }
    .sidebar-logo {
        display: block;
        margin: 0 auto 1.5rem auto;
        padding: 0.5rem;
    }
    </style>
""", unsafe_allow_html=True)

# ------------------------
# Sidebar
# ------------------------
st.sidebar.image("logo3.png", width=200, caption="AI-Powered Crop Protection")
st.sidebar.title("AgriGuard AI")
app_mode = st.sidebar.radio("Navigate", ["Home", "About", "Crop Disease Recognition"])
st.sidebar.markdown("---")
st.sidebar.info("ℹ️ Upload plant leaf images for quick disease diagnosis")

# ------------------------
# Home Page
# ------------------------
if app_mode == "Home":
    st.header("🌿 Smart Crop Disease Recognition System")
    st.markdown("---")
    
    col1, col2, col3 = st.columns([1, 6, 1])
    with col2:
        st.image("homeIMG.jpg", use_container_width=True, caption="Healthy Crops, Better Harvest")
    
    st.markdown("""
    ### Welcome to Agricultural AI Guardian!
    **Our mission**: Empower farmers with instant plant disease detection using AI. 
    
    🚀 **How It Works**
    1. **Capture** - Take a clear photo of the suspect plant leaf
    2. **Upload** - Visit **Crop Disease Recognition** page to submit your image
    3. **Analyze** - Our AI processes the image using deep learning
    4. **Results** - Get instant diagnosis and management tips

    ✨ **Key Benefits**
    - 🎯 High Accuracy: State-of-the-art CNNs
    - ⚡ Real-time Results
    - 🌍 Multiple Plant Varieties Supported
    - 📱 Mobile-friendly
    """)

# ------------------------
# About Page
# ------------------------
elif app_mode == "About":
    st.header("📚 About This Project")
    st.markdown("---")
    
    with st.expander("🌐 Project Overview", expanded=True):
        st.markdown("""
        This AI-powered solution helps farmers quickly identify plant diseases through leaf image analysis, 
        enabling early intervention and reducing crop losses.
        """)
    
    with st.expander("📊 Dataset Information"):
        st.markdown("""
        #### Original Dataset
        - Source: [Plant Diseases Dataset](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset)
        - 87,000+ RGB images
        - 38 plant disease classes
        
        #### Our Implementation
        - Training Split: 80%
        - Validation Split: 20%
        - Test Set: 33 curated real-world images
        - Augmentation: Rotation, flipping, zoom
        """)
    
    with st.expander("🛠️ Technical Architecture"):
        st.markdown("""
        - **Framework**: TensorFlow 2.x
        - **Model**: Custom CNN
        - **Training**: 50 epochs, Adam optimizer
        - **Accuracy**: 98.7% validation accuracy
        - **Inference**: GPU-accelerated
        """)
    
    st.write("© 2025 AgriGuard AI | Developed with ❤️ Shivani Jangam")

# ------------------------
# Crop Disease Recognition Page
# ------------------------
elif app_mode == "Crop Disease Recognition":
    st.header("Crop Disease Analysis 🔍")
    st.markdown("---")
    
    st.subheader("📤 Step 1: Upload Leaf Image")
    test_image = st.file_uploader("Choose a plant leaf image:", type=["jpg", "png", "jpeg"])
    
    if test_image:
        st.subheader("Image Preview 📷")
        st.image(test_image, use_container_width=True, caption="Uploaded Leaf Image")
        
        st.subheader("Step 2: Disease Diagnosis")
        if st.button("Start Analysis 🚀"):
            with st.spinner("Analyzing leaf patterns..."):
                result_index = model_prediction(test_image)
                
                class_name = [
                    'Apple - Apple Scab', 'Apple - Black Rot', 'Apple - Cedar Apple Rust', 'Apple - Healthy',
                    'Blueberry - Healthy', 'Cherry - Powdery Mildew', 'Cherry - Healthy',
                    'Corn - Cercospora Leaf Spot', 'Corn - Common Rust', 'Corn - Northern Leaf Blight', 'Corn - Healthy',
                    'Grape - Black Rot', 'Grape - Esca (Black Measles)', 'Grape - Leaf Blight', 'Grape - Healthy',
                    'Orange - Huanglongbing (Citrus Greening)', 'Peach - Bacterial Spot', 'Peach - Healthy',
                    'Bell Pepper - Bacterial Spot', 'Bell Pepper - Healthy', 'Potato - Early Blight',
                    'Potato - Late Blight', 'Potato - Healthy', 'Raspberry - Healthy', 'Soybean - Healthy',
                    'Squash - Powdery Mildew', 'Strawberry - Leaf Scorch', 'Strawberry - Healthy',
                    'Tomato - Bacterial Spot', 'Tomato - Early Blight', 'Tomato - Late Blight',
                    'Tomato - Leaf Mold', 'Tomato - Septoria Leaf Spot', 'Tomato - Spider Mites',
                    'Tomato - Target Spot', 'Tomato - Yellow Leaf Curl Virus', 'Tomato - Mosaic Virus', 'Tomato - Healthy'
                ]
                
                diagnosis = class_name[result_index]
                plant, disease = diagnosis.split(" - ")
                
                if "Healthy" in disease:
                    st.success(f"🎉 Great news! This {plant.lower()} plant appears healthy!")
                else:
                    st.error(f"⚠️ Alert: Potential {disease} detected in {plant.lower()}!")
                
                st.markdown(f"""
                <div class="prediction-result">
                    <h3 style="color:#2e7d32;">Plant: {plant}</h3>
                    <h3 style="color:#d32f2f;">Condition: {disease}</h3>
                </div>
                """, unsafe_allow_html=True)
