# AI-Powered Orbital Object Detection and Classification System

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-orange.svg)](https://github.com/ultralytics/ultralytics)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An advanced AI system that utilizes YOLOv8 neural networks to detect and classify various space objects including satellites, debris, and other space equipment. The system provides real-time detection results with precise bounding boxes, confidence scores, and comprehensive analytics through an interactive web interface.

## 🚀 Features

- **Real-time Detection**: Instant space object detection and classification
- **11 Object Classes**: Comprehensive detection of satellites and space debris
- **Adjustable Confidence**: Customizable detection sensitivity
- **Interactive Web Interface**: User-friendly Streamlit application
- **Detailed Analytics**: Comprehensive detection metrics and visualizations
- **High Accuracy**: 86.5% mAP, 85.1% precision, 76.7% recall

## 🛰️ Detectable Objects

### Satellites & Spacecraft
- CHEOPS (Characterising Exoplanet Satellite)
- Earth Observation Satellite
- LISA Pathfinder
- PROBA-2 & PROBA-3 (CSC/OCS)
- SMART-1
- SOHO (Solar and Heliospheric Observatory)
- XMM-Newton

### Space Debris
- Various debris types
- Double start objects

## 🔧 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/AI-Powered-Orbital-Object-Detection.git
   cd AI-Powered-Orbital-Object-Detection
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download the trained model**
   - Ensure `best.pt` is in the root directory
   - Or train your own model using the provided scripts

## 🚀 Usage

### Web Application

1. **Start the Streamlit application**
   ```bash
   streamlit run app.py
   ```

2. **Open your browser** and navigate to `http://localhost:8501`

3. **Upload an image** containing space objects

4. **Adjust detection settings** in the sidebar

5. **Click "Analyze Image"** to get detection results

## 📊 Performance Metrics

| Metric | Score |
|--------|-------|
| mAP50 | 86.5% |
| Precision | 85.1% |
| Recall | 76.7% |
| F1-Score | 79.2% |

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- **YOLOv8**: Ultralytics team for the amazing YOLO framework
- **Streamlit**: For the intuitive web application framework
- **Space Agencies**: For inspiring this space safety initiative

---

**⭐ Star this repository if you found it helpful!**


