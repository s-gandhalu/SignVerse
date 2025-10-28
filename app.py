import os
import sys
import warnings

# CRITICAL: Set BEFORE any PyTorch imports
os.environ['TORCH_FORCE_WEIGHTS_ONLY_LOAD'] = '0'
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

# Suppress specific warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

import torch

# MONKEY PATCH: Override torch.load to always use weights_only=False
original_torch_load = torch.load

def patched_torch_load(f, *args, **kwargs):
    """Patched version that forces weights_only=False"""
    kwargs['weights_only'] = False
    return original_torch_load(f, *args, **kwargs)

# Replace torch.load with our patched version
torch.load = patched_torch_load

print("✅ PyTorch patched to disable weights_only restriction")

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import cv2
import numpy as np
import base64
from collections import deque, Counter
from ultralytics import YOLO

app = Flask(__name__)
CORS(app)

# --- MODEL LOADING & CONFIGURATION ---
MODEL_PATH = 'best.pt'
model = None

print("🔄 Loading model...")
try:
    model = YOLO(MODEL_PATH)
    
    if model is not None:
        dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)
        _ = model(dummy_img, verbose=False)
        print(f"✅ Model loaded and verified successfully from {MODEL_PATH}")
    else:
        print(f"❌ Model object is None")
        
except Exception as e:
    print(f"❌ Error loading model: {e}")
    import traceback
    traceback.print_exc()
    model = None

# Class names (39 classes: 13 gestures + 26 letters)
CLASS_NAMES = {
    0: 'Eat', 1: 'Father', 2: 'Fine', 3: 'Hello', 4: 'Help',
    5: 'I Love You', 6: 'I (Me)', 7: 'Mother', 8: 'No', 9: 'Please',
    10: 'Thank You', 11: 'What?', 12: 'Yes',
    13: 'A', 14: 'B', 15: 'C', 16: 'D', 17: 'E', 18: 'F',
    19: 'G', 20: 'H', 21: 'I', 22: 'J', 23: 'K', 24: 'L',
    25: 'M', 26: 'N', 27: 'O', 28: 'P', 29: 'Q', 30: 'R',
    31: 'S', 32: 'T', 33: 'U', 34: 'V', 35: 'W', 36: 'X',
    37: 'Y', 38: 'Z'
}

# Sign language descriptions for text-to-sign conversion
SIGN_DESCRIPTIONS = {
    'A': 'Make a fist with thumb on the side',
    'B': 'Flat hand with fingers together, thumb across palm',
    'C': 'Curved hand forming a C shape',
    'D': 'Point index finger up, other fingers touch thumb',
    'E': 'Curl all fingers down onto thumb',
    'F': 'Touch index and thumb tips, other fingers up',
    'G': 'Point index and thumb horizontally',
    'H': 'Point index and middle fingers horizontally',
    'I': 'Pinky finger up, other fingers down',
    'J': 'Pinky finger draws a J in the air',
    'K': 'Index and middle fingers up in V, thumb touches middle finger',
    'L': 'Index and thumb form L shape',
    'M': 'Tuck thumb under first three fingers',
    'N': 'Tuck thumb under first two fingers',
    'O': 'Fingertips touch thumb forming O',
    'P': 'Like K but pointing down',
    'Q': 'Like G but pointing down',
    'R': 'Cross index and middle fingers',
    'S': 'Make a fist with thumb across fingers',
    'T': 'Thumb between index and middle finger',
    'U': 'Index and middle fingers together pointing up',
    'V': 'Index and middle fingers apart in V shape',
    'W': 'Index, middle, and ring fingers up',
    'X': 'Bent index finger',
    'Y': 'Thumb and pinky extended',
    'Z': 'Draw Z shape with index finger',
    'Hello': 'Open hand, palm out, move from forehead outward',
    'Thank You': 'Fingertips on chin, move hand forward',
    'Please': 'Flat hand on chest, circular motion',
    'Yes': 'Fist nods like a head',
    'No': 'Index and middle fingers snap shut',
    'Help': 'Fist on palm, lift up together',
    'I (Me)': 'Point to yourself with index finger',
    'I Love You': 'Thumb, index, and pinky extended',
    'Mother': 'Thumb on chin, hand open',
    'Father': 'Thumb on forehead, hand open',
    'Eat': 'Fingertips to mouth',
    'Fine': 'Thumb and index finger touch, move hand',
    'What?': 'Shake both hands with palms up'
}

# --- OPTIMIZED TEMPORAL LOGIC ---
BUFFER_SIZE = 8  # Reduced from 15 for faster response
STABILITY_THRESHOLD = 6  # Reduced from 12 for quicker confirmation
MIN_CONFIDENCE_FOR_SMOOTHING = 0.45  # Slightly increased for accuracy

detection_history = deque(maxlen=BUFFER_SIZE)
confidence_history = deque(maxlen=BUFFER_SIZE)
current_sentence = ""
current_word = ""
last_confirmed_id = -2

# --- UTILITY FUNCTIONS ---

def frame_to_base64(frame):
    """Convert frame to base64 for transmission"""
    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return base64.b64encode(buffer).decode('utf-8')

def process_frame(frame, confidence_threshold=0.5):
    """Process a frame and return detections with enhanced annotations."""
    
    if model is None:
        error_frame = frame.copy()
        cv2.putText(error_frame, "MODEL NOT LOADED", (50, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        return error_frame, []
    
    try:
        conf_thresh = max(0.35, min(1.0, float(confidence_threshold)))
    except (ValueError, TypeError):
        conf_thresh = 0.5
        
    try:
        # Optimized inference settings
        results = model(frame, conf=conf_thresh, verbose=False, imgsz=640, half=False)
        detections = []
        annotated_frame = frame.copy()
        
        for result in results:
            if len(result.boxes) == 0:
                continue
                
            # Get top 3 detections
            sorted_indices = result.boxes.conf.argsort(descending=True)[:3]
            
            for rank, idx in enumerate(sorted_indices):
                box = result.boxes[idx]
                conf = float(box.conf[0])
                class_id = int(box.cls[0])
                class_name = CLASS_NAMES.get(class_id, 'Unknown')
                
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                detection = {
                    'class': class_name,
                    'class_id': class_id,
                    'confidence': round(conf, 3),
                    'box': [x1, y1, x2, y2],
                    'rank': rank
                }
                detections.append(detection)
                
                # Enhanced visualization
                if rank == 0:
                    color = (0, 255, 100)  # Bright green for primary
                    thickness = 3
                elif rank == 1:
                    color = (100, 200, 255)  # Light blue for secondary
                    thickness = 2
                else:
                    color = (150, 150, 150)  # Gray for tertiary
                    thickness = 2
                
                # Draw rounded rectangle effect
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, thickness)
                
                # Enhanced label with confidence bar
                label = f"{class_name}"
                conf_text = f"{conf:.0%}"
                
                (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                
                # Draw label background with slight transparency effect
                overlay = annotated_frame.copy()
                cv2.rectangle(overlay, (x1, y1 - label_h - 15), (x1 + max(label_w, 100), y1), color, -1)
                cv2.addWeighted(overlay, 0.7, annotated_frame, 0.3, 0, annotated_frame)
                
                # Draw text
                cv2.putText(annotated_frame, label, (x1 + 5, y1 - 8),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                
                # Confidence bar
                bar_width = int(80 * conf)
                cv2.rectangle(annotated_frame, (x1 + 5, y1 - 5), (x1 + 5 + bar_width, y1 - 2), (255, 255, 255), -1)
        
        return annotated_frame, detections
    
    except Exception as e:
        print(f"❌ Error during inference: {e}")
        error_frame = frame.copy()
        cv2.putText(error_frame, f"ERROR: {str(e)[:30]}", (50, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return error_frame, []

def process_temporal_logic(current_id, current_confidence):
    """
    Enhanced temporal smoothing with confidence weighting.
    """
    global last_confirmed_id, current_sentence, current_word, detection_history, confidence_history
    
    # Add to history with confidence weighting
    if current_confidence >= MIN_CONFIDENCE_FOR_SMOOTHING:
        detection_history.append(current_id)
        confidence_history.append(current_confidence)
    else:
        detection_history.append(-1)
        confidence_history.append(0)

    # Need minimum frames before confirmation
    if len(detection_history) < BUFFER_SIZE:
        return {
            'status': 'pending',
            'word': current_word,
            'sentence': current_sentence,
            'last_sign': None,
            'confidence': current_confidence
        }

    # Weighted voting based on confidence
    class_scores = {}
    for det_id, conf in zip(detection_history, confidence_history):
        if det_id >= 0:
            class_scores[det_id] = class_scores.get(det_id, 0) + conf
    
    if not class_scores:
        return {
            'status': 'stable',
            'word': current_word,
            'sentence': current_sentence,
            'last_sign': None,
            'confidence': 0
        }
    
    # Get best scored class
    best_id = max(class_scores, key=class_scores.get)
    best_score = class_scores[best_id]
    
    # Calculate stability (count of best_id appearances)
    stability_count = sum(1 for x in detection_history if x == best_id)
    
    # Confirm if stable enough AND different from last AND good score
    if (stability_count >= STABILITY_THRESHOLD and 
        best_id != last_confirmed_id and 
        best_score >= MIN_CONFIDENCE_FOR_SMOOTHING * STABILITY_THRESHOLD):
        
        confirmed_name = CLASS_NAMES.get(best_id, 'No_Sign')
        
        # Letter detection (classes 13-38)
        if best_id >= 13:
            current_word += confirmed_name
        else:
            # Gesture/phrase detection
            if current_word:
                current_sentence += current_word + " "
                current_word = ""
            current_sentence += f"{confirmed_name} "

        last_confirmed_id = best_id
        detection_history.clear()
        confidence_history.clear()
        
        return {
            'status': 'confirmed',
            'word': current_word,
            'sentence': current_sentence.strip(),
            'last_sign': confirmed_name,
            'confidence': best_score / len(detection_history)
        }
    
    return {
        'status': 'stable',
        'word': current_word,
        'sentence': current_sentence.strip(),
        'last_sign': None,
        'confidence': best_score / len(detection_history) if class_scores else 0
    }


# --- FLASK ROUTES ---

@app.route('/')
def index():
    """Serve the main HTML page"""
    return render_template('index.html')

@app.route('/api/detect-continuous', methods=['POST'])
def detect_continuous():
    """Processes frame, runs YOLO, and applies temporal smoothing."""
    try:
        data = request.json
        
        # Decode image
        image_data = data['image'].split(',')[1] if ',' in data['image'] else data['image']
        img_array = np.frombuffer(base64.b64decode(image_data), dtype=np.uint8)
        frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        
        if frame is None:
            return jsonify({'error': 'Failed to decode image'}), 400
        
        # Get confidence threshold
        confidence = float(data.get('confidence', 0.5))
        
        # Process frame
        annotated_frame, detections = process_frame(frame, confidence_threshold=confidence)
        
        # Get dominant detection
        if detections:
            dominant_id = detections[0]['class_id']
            dominant_confidence = detections[0]['confidence']
        else:
            dominant_id = -1
            dominant_confidence = 0.0
        
        # Apply temporal logic
        smoothed_state = process_temporal_logic(dominant_id, dominant_confidence)
        
        # Convert annotated frame to base64
        output_image = frame_to_base64(annotated_frame)
        
        return jsonify({
            'status': 'success',
            'image': output_image,
            'raw_detections': detections,
            'smoothed_state': smoothed_state
        })
    
    except Exception as e:
        print(f"❌ Error in /api/detect-continuous: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/text-to-sign', methods=['POST'])
def text_to_sign():
    """Convert text to sign language descriptions."""
    try:
        data = request.json
        text = data.get('text', '').upper().strip()
        
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        
        signs = []
        
        for char in text:
            if char == ' ':
                continue
            elif char in SIGN_DESCRIPTIONS:
                signs.append({
                    'sign': char,
                    'description': SIGN_DESCRIPTIONS[char]
                })
            else:
                # Try to find as word/phrase
                word_match = None
                for key in SIGN_DESCRIPTIONS:
                    if key.upper() == char:
                        word_match = key
                        break
                
                if word_match:
                    signs.append({
                        'sign': word_match,
                        'description': SIGN_DESCRIPTIONS[word_match]
                    })
        
        return jsonify({
            'status': 'success',
            'signs': signs
        })
    
    except Exception as e:
        print(f"❌ Error in /api/text-to-sign: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/clear-state', methods=['POST'])
def clear_state():
    """Resets the global temporal state."""
    global current_sentence, current_word, last_confirmed_id, detection_history, confidence_history
    
    current_sentence = ""
    current_word = ""
    last_confirmed_id = -2
    detection_history.clear()
    confidence_history.clear()
    
    return jsonify({'status': 'cleared'})

@app.route('/api/add-space', methods=['POST'])
def add_space():
    """Manually add a space to the sentence."""
    global current_sentence, current_word
    
    if current_word:
        current_sentence += current_word + " "
        current_word = ""
    else:
        current_sentence += " "
    
    return jsonify({
        'status': 'success',
        'sentence': current_sentence.strip()
    })

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'message': 'SignVerse Backend Running',
        'model_loaded': model is not None,
        'pytorch_version': torch.__version__,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'buffer_size': BUFFER_SIZE,
        'stability_threshold': STABILITY_THRESHOLD
    })

if __name__ == '__main__':
    print(f"\n{'='*60}")
    print(f"🚀 SignVerse Pro Server Starting")
    print(f"{'='*60}")
    print(f"📊 Model Status: {'✅ Loaded' if model else '❌ Failed'}")
    print(f"🎯 Total Classes: 39 (13 gestures + 26 letters)")
    print(f"⚡ Buffer Size: {BUFFER_SIZE} frames (optimized)")
    print(f"🎯 Stability Threshold: {STABILITY_THRESHOLD} frames")
    print(f"🔍 Min Confidence: {MIN_CONFIDENCE_FOR_SMOOTHING}")
    print(f"⚙️  PyTorch Version: {torch.__version__}")
    print(f"🖥️  Device: {'CUDA (GPU)' if torch.cuda.is_available() else 'CPU'}")
    print(f"🌐 Server: http://127.0.0.1:5000")
    print(f"{'='*60}\n")
    
    app.run(debug=False, host='0.0.0.0', port=5000, threaded=True)