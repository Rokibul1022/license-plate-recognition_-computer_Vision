import streamlit as st
import sys
import os
import warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from detection.detector import PlateDetector
from vlm.trainer import VLMTrainer
from database.registry import VehicleRegistry
from verification.engine import VerificationEngine
from video_pipeline.processor import VideoProcessor
import cv2
from PIL import Image
import tempfile
import torch

st.set_page_config(page_title="BD License Plate Recognition System", layout="wide")

st.title("🚗 Bangladeshi License Plate Recognition & Vehicle Information Retrieval")
st.markdown("**DISCLAIMER:** All personal data used in this system are synthetically generated for academic research.")

# Initialize models (cached)
@st.cache_resource
def load_models():
    import os
    detector = None
    recognizer = None
    registry = None
    verifier = None
    
    # Load YOLO detector
    yolo_path = "outputs/detection/plate_detector/weights/best.pt"
    if os.path.exists(yolo_path):
        try:
            detector = PlateDetector(yolo_path)
        except Exception as e:
            st.error(f"Error loading YOLO: {e}")
    
    # Load TrOCR System (TrOCR + Attribute Classifier)
    trocr_path = "outputs/trocr/plate_reader"
    classifier_path = "outputs/trocr/attribute_classifier.pth"
    if os.path.exists(trocr_path) and os.path.exists(classifier_path):
        try:
            from vlm.trocr_system import TrOCRSystem
            recognizer = TrOCRSystem(trocr_path, classifier_path)
            st.sidebar.success("TrOCR System Loaded!")
        except Exception as e:
            st.warning(f"TrOCR not ready: {e}")
    
    # Load database
    db_path = "database/vehicle_registry.db"
    if os.path.exists(db_path):
        try:
            registry = VehicleRegistry(db_path)
            verifier = VerificationEngine(registry)
        except Exception as e:
            st.error(f"Error loading database: {e}")
    
    return detector, recognizer, registry, verifier

detector, recognizer, registry, verifier = load_models()

# Show status
st.sidebar.header("System Status")
st.sidebar.write(f"✅ YOLO Detector: {'Loaded' if detector else '❌ Not Found'}")
st.sidebar.write(f"{'✅' if recognizer else '⏳'} TrOCR System: {'Loaded' if recognizer else 'Not Trained'}")
st.sidebar.write(f"✅ Database: {'Loaded' if registry else '❌ Not Found'}")

# Sidebar
st.sidebar.header("Options")
mode = st.sidebar.selectbox("Select Mode", ["Image Upload", "Video Upload", "Database Query"])

if mode == "Image Upload":
    st.header("Upload Vehicle Image")
    
    uploaded_file = st.file_uploader("Choose an image", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file and detector and recognizer:
        # Display image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # Save temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name
        
        if st.button("Detect & Recognize"):
            with st.spinner("Processing..."):
                # Detect
                detections = detector.detect(tmp_path)
                
                if detections:
                    det = detections[0]
                    
                    # Crop plate for text recognition
                    plate_img = detector.crop_plate(tmp_path, det['plate_bbox'])
                    
                    # Load full vehicle image for attribute classification
                    from PIL import Image as PILImage
                    vehicle_img = PILImage.open(tmp_path).convert('RGB')
                    
                    # Use EasyOCR for reliable plate reading
                    from baselines.ocr_baseline import OCRBaseline
                    ocr = OCRBaseline()
                    ocr_result = ocr.recognize(plate_img)
                    
                    # Get attributes from trained classifier
                    try:
                        attributes = recognizer.classifier.predict(vehicle_img)
                        result = {
                            'plate': ocr_result['plate'],
                            'color': attributes['color'],
                            'type': attributes['type']
                        }
                        method = "EasyOCR + Trained Classifier"
                    except:
                        result = ocr_result
                        method = "EasyOCR Baseline"
                    
                    # Display results
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("Detection Result")
                        st.write(f"**Confidence:** {det['confidence']:.2f}")
                        st.write(f"**Method:** {method}")
                        st.image(plate_img, caption="Detected Plate")
                    
                    with col2:
                        st.subheader("Recognition Result")
                        st.write(f"**Plate:** {result.get('plate', 'N/A')}")
                        st.write(f"**Color:** {result.get('color', 'N/A')}")
                        st.write(f"**Type:** {result.get('type', 'N/A')}")
                    
                    # Verification
                    if verifier:
                        verification = verifier.verify(
                            result.get('plate', ''),
                            result.get('color', ''),
                            result.get('type', '')
                        )
                        
                        st.subheader("Verification Status")
                        if verification['status'] == 'suspicious':
                            st.error(f" SUSPICIOUS VEHICLE")
                            for flag in verification['flags']:
                                st.warning(flag)
                        else:
                            st.success(" Verified")
                        
                        # Information Retrieval
                        if verification['db_info']:
                            st.subheader("Vehicle Information Retrieval")
                            
                            info_type = st.selectbox(
                                "Select Information Type",
                                ["Personal Information", "Vehicle Information", "Plate Details", "Last Known Location", "Movement History"]
                            )
                            
                            db_info = verification['db_info']
                            
                            if info_type == "Personal Information":
                                st.json(db_info['owner'])
                            
                            elif info_type == "Vehicle Information":
                                st.write(f"**Type:** {db_info['type']}")
                                st.write(f"**Color:** {db_info['color']}")
                                st.write(f"**Plate:** {db_info['plate_number']}")
                            
                            elif info_type == "Plate Details":
                                if 'plate_details' in db_info:
                                    st.json(db_info['plate_details'])
                            
                            elif info_type == "Last Known Location":
                                if 'last_location' in db_info:
                                    st.json(db_info['last_location'])
                            
                            elif info_type == "Movement History":
                                history = registry.get_movement_history(db_info['plate_number'])
                                st.json(history)
                
                else:
                    st.error("No vehicle detected in image")

elif mode == "Video Upload":
    st.header("Upload Video for Processing")
    
    uploaded_video = st.file_uploader("Choose a video", type=['mp4', 'avi', 'mov'])
    
    if uploaded_video and detector and recognizer and verifier:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
            tmp.write(uploaded_video.getvalue())
            tmp_path = tmp.name
        
        if st.button("Process Video"):
            with st.spinner("Processing video... This may take a while"):
                processor = VideoProcessor(detector, recognizer, verifier)
                results = processor.process_video(tmp_path)
                
                st.success(f"Processed {len(results)} detections")
                
                # Display suspicious vehicles
                suspicious = [r for r in results if r['verification']['status'] == 'suspicious']
                
                if suspicious:
                    st.subheader(f" {len(suspicious)} Suspicious Vehicles Detected")
                    for r in suspicious:
                        with st.expander(f"Plate: {r['recognition'].get('plate', 'Unknown')}"):
                            st.json(r)

elif mode == "Database Query":
    st.header("Query Vehicle Registry")
    
    if registry:
        plate_number = st.text_input("Enter Plate Number")
        
        if st.button("Search"):
            if plate_number:
                result = registry.query(plate_number)
                
                if result:
                    st.success("Vehicle Found")
                    
                    tab1, tab2, tab3, tab4 = st.tabs(["Owner Info", "Vehicle Info", "Plate Details", "Movement History"])
                    
                    with tab1:
                        st.json(result['owner'])
                    
                    with tab2:
                        st.write(f"**Type:** {result['type']}")
                        st.write(f"**Color:** {result['color']}")
                    
                    with tab3:
                        if 'plate_details' in result:
                            st.json(result['plate_details'])
                    
                    with tab4:
                        history = registry.get_movement_history(plate_number)
                        st.json(history)
                else:
                    st.error("Vehicle not found in registry")

# Footer
st.sidebar.markdown("---")
st.sidebar.info("""
**System Components:**
- YOLOv8 Detection
- PaliGemma 3B VLM
- QLoRA Fine-tuning
- Synthetic Database
- Context Verification
""")
