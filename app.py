import streamlit as st
import os
import sys
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import yaml
from datetime import datetime
import tempfile
import io
import base64

# Add src directory to path for imports
sys.path.append('src')

# Page configuration
st.set_page_config(
    page_title="Space Debris Detection System",
    layout="wide",
    initial_sidebar_state="expanded"
)

try:
    from ultralytics import YOLO
    print("✅ YOLO imported successfully")
except ImportError as e:
    st.error(f"❌ YOLO not found. Error: {e}")
    st.error("Please install ultralytics: `pip install ultralytics`")
    st.stop()
except Exception as e:
    st.error(f"❌ Unexpected error importing YOLO: {e}")
    st.stop()

# Custom CSS for better styling
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global styles */
    .main {
        font-family: 'Inter', sans-serif;
        color: #2d3748;
    }
    
    /* Main header */
    .main-header {
        font-size: 3.5rem;
        font-weight: 700;
        text-align: center;
        color: #2d3748;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Subtitle */
    .subtitle {
        text-align: center;
        font-size: 1.2rem;
        color: #4a5568;
        margin-bottom: 3rem;
        font-weight: 400;
    }
    
    /* Info boxes */
    .info-box {
        background: #ffffff;
        padding: 1.5rem;
        border-radius: 15px;
        border: 2px solid #e2e8f0;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        margin: 1.5rem 0;
        color: #2d3748;
    }
    
    .info-box h3, .info-box h4 {
        color: #2d3748;
        margin-bottom: 1rem;
    }
    
    .info-box p {
        color: #4a5568;
        line-height: 1.6;
    }
    
    /* Detection result box */
    .detection-result {
        background: #ffffff;
        padding: 1.5rem;
        border-radius: 15px;
        border: 2px solid #2196f3;
        margin: 1.5rem 0;
        box-shadow: 0 8px 24px rgba(33, 150, 243, 0.2);
        color: #2d3748;
    }
    
    /* Success box */
    .success-box {
        background: #ffffff;
        padding: 1.5rem;
        border-radius: 15px;
        border: 2px solid #4caf50;
        margin: 1.5rem 0;
        box-shadow: 0 8px 24px rgba(76, 175, 80, 0.2);
        color: #2d3748;
    }
    
    /* Warning box */
    .warning-box {
        background: #ffffff;
        padding: 1.5rem;
        border-radius: 15px;
        border: 2px solid #ff9800;
        margin: 1.5rem 0;
        box-shadow: 0 8px 24px rgba(255, 152, 0, 0.2);
        color: #2d3748;
    }
    
    /* Error box */
    .error-box {
        background: #ffffff;
        padding: 1.5rem;
        border-radius: 15px;
        border: 2px solid #f44336;
        margin: 1.5rem 0;
        box-shadow: 0 8px 24px rgba(244, 67, 54, 0.2);
        color: #2d3748;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border: none;
        border-radius: 25px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1.1rem;
        color: white;
        box-shadow: 0 8px 24px rgba(102, 126, 234, 0.3);
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 32px rgba(102, 126, 234, 0.4);
        color: white;
    }
    
    /* Metrics styling */
    .metric-card {
        background: #ffffff;
        padding: 1rem;
        border-radius: 12px;
        border: 2px solid #e2e8f0;
        box-shadow: 0 4px 16px rgba(0,0,0,0.05);
        text-align: center;
        margin: 0.5rem 0;
        color: #2d3748;
    }
    
    .metric-card strong {
        color: #2d3748;
    }
    
    /* Section headers */
    .section-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #2d3748;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid #667eea;
    }
    
    /* Footer */
    .footer {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        text-align: center;
        padding: 2rem;
        border-radius: 15px;
        margin-top: 3rem;
    }
    
    .footer h3, .footer p {
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# App title and description
st.markdown('<h1 class="main-header">Space Debris Detection System</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">AI-Powered Space Object Detection & Classification</p>', unsafe_allow_html=True)

# st.markdown("""
# <div class="info-box">
# <h3>About this Application</h3>
# <p>This cutting-edge AI system utilizes advanced YOLO neural networks to detect and classify various space objects including satellites, debris, and other space equipment. Simply upload an image to get real-time detection results with precise bounding boxes and confidence scores.</p>
# <p><strong>Features:</strong> Real-time detection • 11 object classes • Adjustable confidence • Detailed analytics</p>
# </div>
# """, unsafe_allow_html=True)

# Sidebar configuration
st.sidebar.markdown("### Detection Settings")
st.sidebar.markdown("---")

confidence_threshold = st.sidebar.slider(
    "Confidence Threshold", 
    min_value=0.1, 
    max_value=1.0, 
    value=0.5, 
    step=0.05,
    help="Minimum confidence level for detections. Lower values detect more objects but may include false positives."
)

# Model path selection
model_options = []
if os.path.exists('best.pt'):
    model_options.append('best.pt')
if os.path.exists('runs/train/exp/weights/best.pt'):
    model_options.append('runs/train/exp/weights/best.pt')

if not model_options:
    st.markdown('<div class="error-box"><strong>No trained model found!</strong><br>Please ensure "best.pt" is in the root directory or train a model first.</div>', unsafe_allow_html=True)
    st.stop()

selected_model = st.sidebar.selectbox("Select Model", model_options, help="Choose the trained model for detection")

# Load class names
@st.cache_data
def load_class_names():
    try:
        with open('data.yaml', 'r') as f:
            data_config = yaml.safe_load(f)
        return data_config['names']
    except FileNotFoundError:
        st.warning("data.yaml not found. Using default class names.")
        return ['cheops', 'debris', 'double_start', 'earth_observation_sat_1', 'lisa_pathfinder', 
                'proba_2', 'proba_3_csc', 'proba_3_ocs', 'smart_1', 'soho', 'xmm_newton']

# Load model
@st.cache_resource
def load_model(model_path):
    try:
        model = YOLO(model_path)
        return model
    except Exception as e:
        st.error(f"Failed to load model: {str(e)}")
        return None

# Detection function
def detect_space_debris(image, model, classes, confidence_threshold):
    """Perform space debris detection on uploaded image"""
    
    # Save uploaded image temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
        image.save(tmp_file.name)
        temp_path = tmp_file.name
    
    try:
        # Run detection
        results = model(temp_path, conf=confidence_threshold)[0]
        
        # Create detection visualization
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.imshow(image)
        
        # Define custom vibrant colors for different classes
        custom_colors = [
            '#FF6B6B',  # Red
            '#4ECDC4',  # Teal
            '#45B7D1',  # Blue
            '#96CEB4',  # Green
            '#FECA57',  # Yellow
            '#FF9FF3',  # Pink
            '#54A0FF',  # Light Blue
            '#5F27CD',  # Purple
            '#00D2D3',  # Cyan
            '#FF9F43',  # Orange
            '#10AC84'   # Emerald
        ]
        
        # Ensure we have enough colors
        while len(custom_colors) < len(classes):
            custom_colors.extend(custom_colors)
        
        detections = []
        
        # Draw predictions
        for box in results.boxes:
            # Get box coordinates
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            confidence = box.conf[0].cpu().numpy()
            class_id = int(box.cls[0].cpu().numpy())
            
            # Store detection info
            detections.append({
                'class': classes[class_id],
                'confidence': confidence,
                'bbox': [x1, y1, x2, y2]
            })
            
            # Create rectangle patch with custom styling
            rect = patches.Rectangle(
                (x1, y1), x2-x1, y2-y1, 
                linewidth=4, 
                edgecolor=custom_colors[class_id], 
                facecolor='none', 
                alpha=0.9,
                linestyle='-'
            )
            ax.add_patch(rect)
            
            # Add a subtle shadow effect
            shadow_rect = patches.Rectangle(
                (x1+2, y1+2), x2-x1, y2-y1, 
                linewidth=4, 
                edgecolor='black', 
                facecolor='none', 
                alpha=0.3,
                linestyle='-'
            )
            ax.add_patch(shadow_rect)
            
            # Add label with improved styling
            label = f'{classes[class_id].replace("_", " ").title()}: {confidence:.1%}'
            ax.text(
                x1, y1-15, label, 
                color='white',
                fontsize=11, 
                fontweight='bold',
                bbox=dict(
                    facecolor=custom_colors[class_id], 
                    alpha=0.9, 
                    pad=4,
                    boxstyle="round,pad=0.3"
                ),
                ha='left'
            )
        
        ax.set_xlim(0, image.width)
        ax.set_ylim(image.height, 0)
        ax.axis('off')
        plt.tight_layout()
        
        # Convert plot to image
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, dpi=150)
        buf.seek(0)
        plt.close()
        
        return detections, buf
        
    finally:
        # Clean up temporary file
        os.unlink(temp_path)

# Main application
def main():
    # Load class names and model
    classes = load_class_names()
    model = load_model(selected_model)
    
    if model is None:
        st.stop()
    
    # Sidebar info
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Detectable Objects")
    
    # Display classes in a more organized way
    col1, col2 = st.sidebar.columns(2)
    for i, class_name in enumerate(classes):
        if i % 2 == 0:
            col1.markdown(f"• **{class_name.replace('_', ' ').title()}**")
        else:
            col2.markdown(f"• **{class_name.replace('_', ' ').title()}**")
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Model Info")
    st.sidebar.info(f"**Model:** {selected_model}\\n**Classes:** {len(classes)}\\n**Confidence:** {confidence_threshold:.1%}")
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Tips")
    st.sidebar.markdown("""
    - **Higher confidence** = fewer but more accurate detections
    - **Lower confidence** = more detections but may include false positives
    - **Best images:** Clear, well-lit space objects
    - **Supported formats:** PNG, JPG, JPEG
    """)
    
    # File upload
    st.markdown('<h3 class="section-header">Upload Image for Detection</h3>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Upload Image", 
        type=['png', 'jpg', 'jpeg'],
        help="Upload an image containing space objects for detection",
        label_visibility="collapsed"
    )
    
    # Add some helpful text
    st.markdown("""
    <div style="text-align: center; color: #4a5568; margin: 1rem 0;">
        <p>📁 Drag and drop your image here, or click to browse</p>
        <p><small>Supported formats: PNG, JPG, JPEG | Max file size: 200MB</small></p>
    </div>
    """, unsafe_allow_html=True)
    
    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file).convert('RGB')
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📷 Original Image")
            st.image(image, caption=f"📁 {uploaded_file.name}", use_container_width=True)
            
            # Image info
            st.markdown(f"""
            <div class="metric-card">
                <strong>Dimensions:</strong> {image.width} × {image.height} pixels<br>
                <strong>File Size:</strong> {len(uploaded_file.getvalue()) / 1024:.1f} KB<br>
                <strong>Mode:</strong> {image.mode}
            </div>
            """, unsafe_allow_html=True)
        
        # Detection button with improved styling
        st.markdown("<br>", unsafe_allow_html=True)
        detect_col1, detect_col2, detect_col3 = st.columns([1, 2, 1])
        with detect_col2:
            if st.button("Analyze Image", type="primary", use_container_width=True):
                with st.spinner("AI is analyzing your image..."):
                    detections, result_image_buf = detect_space_debris(
                        image, model, classes, confidence_threshold
                    )
                
                with col2:
                    st.markdown("#### Detection Results")
                    result_image = Image.open(result_image_buf)
                    st.image(result_image, caption="Detection Results", use_container_width=True)
                
                # Display detection summary
                st.markdown("---")
                st.markdown('<h3 class="section-header">Detection Analysis</h3>', unsafe_allow_html=True)
                
                if detections:
                    # Create metrics row
                    met_col1, met_col2, met_col3, met_col4 = st.columns(4)
                    with met_col1:
                        st.metric("Objects Found", len(detections))
                    with met_col2:
                        avg_confidence = sum(det['confidence'] for det in detections) / len(detections)
                        st.metric("Avg Confidence", f"{avg_confidence:.1%}")
                    with met_col3:
                        unique_classes = len(set(det['class'] for det in detections))
                        st.metric("Unique Classes", unique_classes)
                    with met_col4:
                        high_conf = sum(1 for det in detections if det['confidence'] > 0.8)
                        st.metric("High Confidence", f"{high_conf}/{len(detections)}")
                    
                    st.markdown('<div class="success-box">', unsafe_allow_html=True)
                    st.markdown(f"**🎉 Detection Successful!** Found {len(detections)} space objects in your image.")
                    
                    # Create detailed results table
                    st.markdown("#### Detailed Results")
                    detection_data = []
                    for i, det in enumerate(detections, 1):
                        confidence_emoji = "🟢" if det['confidence'] > 0.8 else "🟡" if det['confidence'] > 0.6 else "🔴"
                        detection_data.append({
                            '#': i,
                            'Object Type': det['class'].replace('_', ' ').title(),
                            'Confidence': f"{confidence_emoji} {det['confidence']:.1%}",
                            'Size (W×H)': f"{abs(det['bbox'][2]-det['bbox'][0]):.0f}×{abs(det['bbox'][3]-det['bbox'][1]):.0f}",
                            '📍 Center Position': f"({(det['bbox'][0]+det['bbox'][2])/2:.0f}, {(det['bbox'][1]+det['bbox'][3])/2:.0f})"
                        })
                    
                    st.dataframe(detection_data, use_container_width=True, hide_index=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Class distribution with better visualization
                    st.markdown("#### Object Distribution")
                    class_counts = {}
                    for det in detections:
                        class_name = det['class'].replace('_', ' ').title()
                        class_counts[class_name] = class_counts.get(class_name, 0) + 1
                    
                    # Create visual distribution
                    dist_cols = st.columns(min(len(class_counts), 4))
                    for i, (class_name, count) in enumerate(class_counts.items()):
                        with dist_cols[i % len(dist_cols)]:
                            percentage = (count / len(detections)) * 100
                            st.markdown(f"""
                            <div class="metric-card">
                                <strong style="color: #2d3748;">{class_name}</strong><br>
                                <span style="font-size: 1.5em; color: #667eea;">{count}</span> objects<br>
                                <small style="color: #4a5568;">{percentage:.1f}% of total</small>
                            </div>
                            """, unsafe_allow_html=True)
                
                else:
                    st.markdown('<div class="warning-box">', unsafe_allow_html=True)
                    st.markdown("**🤔 No objects detected** with the current confidence threshold.")
                    st.markdown(f"**Tip:** Try lowering the confidence threshold (currently {confidence_threshold:.1%}) or upload a different image with more visible space objects.")
                    st.markdown('</div>', unsafe_allow_html=True)
    
    else:
        # Show getting started guide when no file is uploaded
        st.markdown('<h3 class="section-header">Getting Started</h3>', unsafe_allow_html=True)
        
        # Feature showcase
        feat_col1, feat_col2, feat_col3, feat_col4 = st.columns(4)
        
        with feat_col1:
            st.markdown("""
            <div class="metric-card">
                <h4 style="color: #2d3748;">Upload</h4>
                <p style="color: #4a5568;">Select an image with space objects</p>
            </div>
            """, unsafe_allow_html=True)
        
        with feat_col2:
            st.markdown("""
            <div class="metric-card">
                <h4 style="color: #2d3748;">Configure</h4>
                <p style="color: #4a5568;">Adjust detection settings</p>
            </div>
            """, unsafe_allow_html=True)
        
        with feat_col3:
            st.markdown("""
            <div class="metric-card">
                <h4 style="color: #2d3748;">Analyze</h4>
                <p style="color: #4a5568;">AI processes your image</p>
            </div>
            """, unsafe_allow_html=True)
        
        with feat_col4:
            st.markdown("""
            <div class="metric-card">
                <h4 style="color: #2d3748;">Results</h4>
                <p style="color: #4a5568;">View detailed detection data</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Detectable objects showcase
        st.markdown('<h3 class="section-header">Detectable Space Objects</h3>', unsafe_allow_html=True)
        
        # Group classes into categories
        satellites = ['cheops', 'earth_observation_sat_1', 'lisa_pathfinder', 'proba_2', 'proba_3_csc', 'proba_3_ocs', 'smart_1', 'soho', 'xmm_newton']
        other_objects = ['debris', 'double_start']
        
        obj_col1, obj_col2 = st.columns(2)
        
        with obj_col1:
            st.markdown("""
            <div class="info-box">
                <h4>Satellites & Spacecraft</h4>
            """, unsafe_allow_html=True)
            for sat in satellites:
                if sat in classes:
                    st.markdown(f"• **{sat.replace('_', ' ').title()}**")
            st.markdown("</div>", unsafe_allow_html=True)
        
        with obj_col2:
            st.markdown("""
            <div class="info-box">
                <h4>🗂️ Other Space Objects</h4>
            """, unsafe_allow_html=True)
            for obj in other_objects:
                if obj in classes:
                    st.markdown(f"• **{obj.replace('_', ' ').title()}**")
            st.markdown("</div>", unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div class="footer">
    <h3>Space Debris Detection System</h3>
    <p><strong>Powered by YOLO Neural Networks & Streamlit</strong></p>
    <p>Advanced AI for Space Safety & Debris Monitoring</p>
    <p><small>Built for space exploration and safety</small></p>
</div>
""", unsafe_allow_html=True)

if __name__ == "__main__":
    main()