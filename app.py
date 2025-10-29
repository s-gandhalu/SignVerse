import os
import sys
import warnings
import time

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

# Enhanced sign language descriptions for text-to-sign conversion
SIGN_DESCRIPTIONS = {
    # Alphabet
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
    
    # Common Greetings & Expressions
    'HELLO': 'Open hand, palm out, move from forehead outward (like a salute)',
    'GOODBYE': 'Wave hand side to side with palm facing out',
    'GOOD MORNING': 'Sign GOOD (thumb up from chin) then MORNING (arm rises like sunrise)',
    'GOOD NIGHT': 'Sign GOOD then NIGHT (arm sets like sunset)',
    
    # Polite Phrases
    'THANK YOU': 'Fingertips on chin, move hand forward and down',
    'PLEASE': 'Flat hand on chest, make circular motion',
    'SORRY': 'Make fist, rub in circle on chest (showing heart)',
    'EXCUSE ME': 'Brush fingertips across opposite palm',
    'YOU\'RE WELCOME': 'Flat hand at chin, move forward (similar to THANK YOU)',
    
    # Questions
    'WHAT?': 'Shake both hands with palms up, eyebrows furrowed',
    'WHERE?': 'Point index finger, shake side to side',
    'WHEN?': 'Circle index fingers around each other, then point',
    'WHY?': 'Touch forehead with middle finger, then move hand away',
    'HOW?': 'Knuckles together, roll hands forward',
    'WHO?': 'Circle index finger around lips',
    
    # Basic Responses
    'YES': 'Make fist, nod it like a head nodding',
    'NO': 'Snap index and middle fingers shut (like mouth closing)',
    'MAYBE': 'Flat hands alternate up and down',
    'I DON\'T KNOW': 'Flat hand waves near head, shrug shoulders',
    
    # Family
    'MOTHER': 'Thumb touches chin, fingers spread (like touching bonnet)',
    'FATHER': 'Thumb touches forehead, fingers spread',
    'SISTER': 'Sign GIRL (thumb on jaw) then point two fingers together',
    'BROTHER': 'Sign BOY (thumb on forehead) then point two fingers together',
    'FAMILY': 'F handshape, make circle connecting hands',
    'BABY': 'Rock arms as if holding a baby',
    'CHILD': 'Pat head height as if patting child\'s head',
    
    # Personal Pronouns
    'I (ME)': 'Point to yourself with index finger on chest',
    'YOU': 'Point index finger toward the person',
    'WE': 'Point to self, then arc finger to include others',
    'THEY': 'Point to others or sweep hand to side',
    
    # Emotions
    'HAPPY': 'Brush hand up chest twice (showing joy rising)',
    'SAD': 'Drag hands down face (like tears falling)',
    'ANGRY': 'Claw hands at face and pull away sharply',
    'LOVE': 'Cross arms over chest, hugging yourself',
    'I LOVE YOU': 'Thumb, index, and pinky extended (combines I, L, Y)',
    
    # Actions
    'EAT': 'Fingertips to mouth (like putting food in)',
    'DRINK': 'Tilt C-shaped hand to mouth (like drinking)',
    'SLEEP': 'Flat hand on cheek, tilt head',
    'HELP': 'Fist on flat palm, lift both up together',
    'WORK': 'Tap S-hands together at wrists',
    'PLAY': 'Y-hands shake back and forth',
    'GO': 'Point index fingers, bend and move forward',
    'COME': 'Index fingers beckon toward you',
    'STOP': 'Flat hand chops down on opposite palm',
    'WAIT': 'Wiggle fingers with both hands in front',
    
    # Common Adjectives
    'GOOD': 'Flat hand at mouth, move down to other hand',
    'BAD': 'Flat hand at mouth, flip away and down',
    'FINE': 'Thumb and fingers spread, tap chest (like showing pride)',
    'BIG': 'Hands start close, move apart',
    'SMALL': 'Flat hands facing, move close together',
    'HOT': 'Claw hand at mouth, twist away quickly',
    'COLD': 'Shiver with S-hands',
    
    # Time
    'NOW': 'Y-hands drop down together',
    'LATER': 'L handshape moves forward',
    'TODAY': 'NOW then DAY signs combined',
    'TOMORROW': 'A-hand on cheek, move forward',
    'YESTERDAY': 'A-hand on cheek, move backward',
    
    # Common Words
    'NAME': 'H-fingers tap on H-fingers (like crossing)',
    'HOME': 'Fingertips touch at mouth, move to cheek',
    'SCHOOL': 'Clap hands twice',
    'FRIEND': 'Hook index fingers, switch positions',
    'WATER': 'W handshape taps chin',
    'FOOD': 'Fingertips to mouth repeatedly',
    
    # Emergency
    'EMERGENCY': 'E-hand shakes urgently',
    'DANGER': 'A-hand strikes up along body',
    'SICK': 'Middle finger touches forehead and stomach',
    'HURT': 'Index fingers twist at location of pain',
    'HOSPITAL': 'H-hand draws cross on upper arm',
    'DOCTOR': 'D-hand on wrist (taking pulse)',
}

# Image URLs for sign language (using free ASL resources)
SIGN_IMAGE_URLS = {
    'A': 'https://www.lifeprint.com/asl101/images-signs/a.jpg',
    'B': 'https://www.lifeprint.com/asl101/images-signs/b.jpg',
    'C': 'https://www.lifeprint.com/asl101/images-signs/c.jpg',
    'D': 'https://www.lifeprint.com/asl101/images-signs/d.jpg',
    'E': 'https://www.lifeprint.com/asl101/images-signs/e.jpg',
    'F': 'https://www.lifeprint.com/asl101/images-signs/f.jpg',
    'G': 'https://www.lifeprint.com/asl101/images-signs/g.jpg',
    'H': 'https://www.lifeprint.com/asl101/images-signs/h.jpg',
    'I': 'https://www.lifeprint.com/asl101/images-signs/i.jpg',
    'J': 'https://www.lifeprint.com/asl101/images-signs/j.jpg',
    'K': 'https://www.lifeprint.com/asl101/images-signs/k.jpg',
    'L': 'https://www.lifeprint.com/asl101/images-signs/l.jpg',
    'M': 'https://www.lifeprint.com/asl101/images-signs/m.jpg',
    'N': 'https://www.lifeprint.com/asl101/images-signs/n.jpg',
    'O': 'https://www.lifeprint.com/asl101/images-signs/o.jpg',
    'P': 'https://www.lifeprint.com/asl101/images-signs/p.jpg',
    'Q': 'https://www.lifeprint.com/asl101/images-signs/q.jpg',
    'R': 'https://www.lifeprint.com/asl101/images-signs/r.jpg',
    'S': 'https://www.lifeprint.com/asl101/images-signs/s.jpg',
    'T': 'https://www.lifeprint.com/asl101/images-signs/t.jpg',
    'U': 'https://www.lifeprint.com/asl101/images-signs/u.jpg',
    'V': 'https://www.lifeprint.com/asl101/images-signs/v.jpg',
    'W': 'https://www.lifeprint.com/asl101/images-signs/w.jpg',
    'X': 'https://www.lifeprint.com/asl101/images-signs/x.jpg',
    'Y': 'https://www.lifeprint.com/asl101/images-signs/y.jpg',
    'Z': 'https://www.lifeprint.com/asl101/images-signs/z.jpg',
    'HELLO': 'https://www.lifeprint.com/asl101/images-signs/hello.jpg',
    'THANK YOU': 'https://www.lifeprint.com/asl101/images-signs/thank_you.jpg',
    'PLEASE': 'https://www.lifeprint.com/asl101/images-signs/please.jpg',
    'YES': 'https://www.lifeprint.com/asl101/images-signs/yes.jpg',
    'NO': 'https://www.lifeprint.com/asl101/images-signs/no.jpg',
    'HELP': 'https://www.lifeprint.com/asl101/images-signs/help.jpg',
    'SORRY': 'https://www.lifeprint.com/asl101/images-signs/sorry.jpg',
    'MOTHER': 'https://www.lifeprint.com/asl101/images-signs/mother.jpg',
    'FATHER': 'https://www.lifeprint.com/asl101/images-signs/father.jpg',
}

# --- ENHANCED TEMPORAL LOGIC WITH LETTER HOLD ---
BUFFER_SIZE = 8
GESTURE_STABILITY_THRESHOLD = 6  # Quick for gestures/phrases
LETTER_STABILITY_MULTIPLIER = 1.5  # Letters need 1.5x more frames
MIN_CONFIDENCE_FOR_SMOOTHING = 0.45

detection_history = deque(maxlen=BUFFER_SIZE * 3)  # Larger buffer for letters
confidence_history = deque(maxlen=BUFFER_SIZE * 3)
current_sentence = ""
current_word = ""
last_confirmed_id = -2
last_confirmation_time = 0
letter_hold_time = 1.5  # Default 1.5 seconds

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
        results = model(frame, conf=conf_thresh, verbose=False, imgsz=640, half=False)
        detections = []
        annotated_frame = frame.copy()
        
        for result in results:
            if len(result.boxes) == 0:
                continue
                
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
                    color = (0, 255, 100)
                    thickness = 3
                elif rank == 1:
                    color = (100, 200, 255)
                    thickness = 2
                else:
                    color = (150, 150, 150)
                    thickness = 2
                
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, thickness)
                
                label = f"{class_name}"
                (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                
                overlay = annotated_frame.copy()
                cv2.rectangle(overlay, (x1, y1 - label_h - 15), (x1 + max(label_w, 100), y1), color, -1)
                cv2.addWeighted(overlay, 0.7, annotated_frame, 0.3, 0, annotated_frame)
                
                cv2.putText(annotated_frame, label, (x1 + 5, y1 - 8),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                
                bar_width = int(80 * conf)
                cv2.rectangle(annotated_frame, (x1 + 5, y1 - 5), (x1 + 5 + bar_width, y1 - 2), (255, 255, 255), -1)
        
        return annotated_frame, detections
    
    except Exception as e:
        print(f"❌ Error during inference: {e}")
        error_frame = frame.copy()
        cv2.putText(error_frame, f"ERROR: {str(e)[:30]}", (50, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return error_frame, []

def process_temporal_logic(current_id, current_confidence, hold_time=1.5):
    """
    Enhanced temporal smoothing with separate handling for letters vs gestures.
    Letters require holding for longer time based on hold_time parameter.
    """
    global last_confirmed_id, current_sentence, current_word
    global detection_history, confidence_history, last_confirmation_time
    
    current_time = time.time()
    
    # Add to history
    if current_confidence >= MIN_CONFIDENCE_FOR_SMOOTHING:
        detection_history.append(current_id)
        confidence_history.append(current_confidence)
    else:
        detection_history.append(-1)
        confidence_history.append(0)

    if len(detection_history) < BUFFER_SIZE:
        return {
            'status': 'pending',
            'word': current_word,
            'sentence': current_sentence,
            'last_sign': None,
            'confidence': current_confidence
        }

    # Weighted voting
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
    
    best_id = max(class_scores, key=class_scores.get)
    best_score = class_scores[best_id]
    stability_count = sum(1 for x in detection_history if x == best_id)
    
    # Determine if this is a letter (class_id >= 13)
    is_letter = best_id >= 13
    
    # Calculate required stability based on type and hold time
    if is_letter:
        # Letters need more stability: base threshold * multiplier * hold_time adjustment
        frames_needed = int(GESTURE_STABILITY_THRESHOLD * LETTER_STABILITY_MULTIPLIER * (hold_time / 1.5))
    else:
        # Gestures use base threshold
        frames_needed = GESTURE_STABILITY_THRESHOLD
    
    # Time-based gate for letters to prevent rapid fire
    time_since_last = current_time - last_confirmation_time
    min_time_between = 0.3 if is_letter else 0.2
    
    if time_since_last < min_time_between:
        return {
            'status': 'cooldown',
            'word': current_word,
            'sentence': current_sentence,
            'last_sign': None,
            'confidence': best_score / len(detection_history)
        }
    
    # Confirm if stable enough AND different from last AND good score
    if (stability_count >= frames_needed and 
        best_id != last_confirmed_id and 
        best_score >= MIN_CONFIDENCE_FOR_SMOOTHING * frames_needed):
        
        confirmed_name = CLASS_NAMES.get(best_id, 'No_Sign')
        
        if is_letter:
            current_word += confirmed_name
        else:
            if current_word:
                current_sentence += current_word + " "
                current_word = ""
            current_sentence += f"{confirmed_name} "

        last_confirmed_id = best_id
        last_confirmation_time = current_time
        detection_history.clear()
        confidence_history.clear()
        
        return {
            'status': 'confirmed',
            'word': current_word,
            'sentence': current_sentence.strip(),
            'last_sign': confirmed_name,
            'confidence': best_score / stability_count,
            'frames_held': stability_count,
            'frames_needed': frames_needed
        }
    
    return {
        'status': 'holding' if stability_count > 0 else 'stable',
        'word': current_word,
        'sentence': current_sentence.strip(),
        'last_sign': None,
        'confidence': best_score / len(detection_history) if class_scores else 0,
        'frames_held': stability_count,
        'frames_needed': frames_needed,
        'progress': min(100, int(stability_count / frames_needed * 100))
    }


# --- FLASK ROUTES ---

@app.route('/')
def index():
    """Serve the main HTML page"""
    return render_template('index.html')

@app.route('/api/detect-continuous', methods=['POST'])
def detect_continuous():
    """Processes frame with enhanced letter hold logic"""
    try:
        data = request.json
        
        # Decode image
        image_data = data['image'].split(',')[1] if ',' in data['image'] else data['image']
        img_array = np.frombuffer(base64.b64decode(image_data), dtype=np.uint8)
        frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        
        if frame is None:
            return jsonify({'error': 'Failed to decode image'}), 400
        
        confidence = float(data.get('confidence', 0.5))
        hold_time = float(data.get('letter_hold_time', 1.5))
        
        annotated_frame, detections = process_frame(frame, confidence_threshold=confidence)
        
        if detections:
            dominant_id = detections[0]['class_id']
            dominant_confidence = detections[0]['confidence']
        else:
            dominant_id = -1
            dominant_confidence = 0.0
        
        smoothed_state = process_temporal_logic(dominant_id, dominant_confidence, hold_time)
        
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
    """Convert text to sign language descriptions with images."""
    try:
        data = request.json
        text = data.get('text', '').upper().strip()
        
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        
        signs = []
        
        # Split into words and process
        words = text.split()
        
        for word in words:
            # Check if whole word is a gesture
            if word in SIGN_DESCRIPTIONS:
                signs.append({
                    'sign': word,
                    'description': SIGN_DESCRIPTIONS[word],
                    'image_url': SIGN_IMAGE_URLS.get(word, None)
                })
            else:
                # Process letter by letter
                for char in word:
                    if char in SIGN_DESCRIPTIONS:
                        signs.append({
                            'sign': char,
                            'description': SIGN_DESCRIPTIONS[char],
                            'image_url': SIGN_IMAGE_URLS.get(char, None)
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
    global current_sentence, current_word, last_confirmed_id
    global detection_history, confidence_history, last_confirmation_time
    
    current_sentence = ""
    current_word = ""
    last_confirmed_id = -2
    last_confirmation_time = 0
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
        'message': 'SignVerse Enhanced Backend Running',
        'model_loaded': model is not None,
        'pytorch_version': torch.__version__,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'buffer_size': BUFFER_SIZE,
        'gesture_threshold': GESTURE_STABILITY_THRESHOLD,
        'letter_multiplier': LETTER_STABILITY_MULTIPLIER
    })

if __name__ == '__main__':
    print(f"\n{'='*60}")
    print(f"🚀 SignVerse Pro Enhanced Server Starting")
    print(f"{'='*60}")
    print(f"📊 Model Status: {'✅ Loaded' if model else '❌ Failed'}")
    print(f"🎯 Total Classes: 39 (13 gestures + 26 letters)")
    print(f"⚡ Buffer Size: {BUFFER_SIZE} frames")
    print(f"🔤 Letter Hold: {LETTER_STABILITY_MULTIPLIER}x longer than gestures")
    print(f"✋ Gesture Threshold: {GESTURE_STABILITY_THRESHOLD} frames")
    print(f"🔍 Min Confidence: {MIN_CONFIDENCE_FOR_SMOOTHING}")
    print(f"⚙️  PyTorch Version: {torch.__version__}")
    print(f"🖥️  Device: {'CUDA (GPU)' if torch.cuda.is_available() else 'CPU'}")
    print(f"🌐 Server: http://127.0.0.1:5000")
    print(f"\n💡 New Features:")
    print(f"   • Letters require longer hold time (adjustable 0.5-3s)")
    print(f"   • Text-to-Sign with visual images")
    print(f"   • Time-based cooldown prevents rapid fire")
    print(f"   • Progress indicator for letter confirmation")
    print(f"{'='*60}\n")
    
    app.run(debug=False, host='0.0.0.0', port=5000, threaded=True)
