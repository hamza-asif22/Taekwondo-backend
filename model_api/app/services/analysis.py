import cv2
import mediapipe as mp
import numpy as np
from scipy.spatial.distance import euclidean
from app.core.config import STANCE_NAMES

# Initialize MediaPipe pose
mp_pose = mp.solutions.pose

# Pose detection configurations
POSE_CONFIGS = {
    'high_accuracy': {'model_complexity': 2, 'min_detection_confidence': 0.8, 'min_tracking_confidence': 0.8},
    'balanced': {'model_complexity': 1, 'min_detection_confidence': 0.7, 'min_tracking_confidence': 0.7},
    'fast': {'model_complexity': 0, 'min_detection_confidence': 0.6, 'min_tracking_confidence': 0.6}
}

# ----------------------------
# 1. Preprocessing and Feature Extraction
# ----------------------------

def preprocess_image(image_path):
    """Load image and apply preprocessing options."""
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"❌ Could not load image: {image_path}")
    return image

def extract_comprehensive_features(image, config='balanced'):
    """Extract comprehensive pose features with multiple preprocessing attempts."""
    pose = mp_pose.Pose(static_image_mode=True, **POSE_CONFIGS[config])

    # Original and contrast-enhanced images
    processed_images = [cv2.cvtColor(image, cv2.COLOR_BGR2RGB)]
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    enhanced = cv2.merge((cl, a, b))
    enhanced_rgb = cv2.cvtColor(cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR), cv2.COLOR_BGR2RGB)
    processed_images.append(enhanced_rgb)

    best_result = None
    max_landmarks = 0

    for img in processed_images:
        results = pose.process(img)
        if results.pose_landmarks:
            landmark_count = len([lm for lm in results.pose_landmarks.landmark if lm.visibility > 0.5])
            if landmark_count > max_landmarks:
                max_landmarks = landmark_count
                best_result = (results, img)

    pose.close()

    if not best_result or not best_result[0].pose_landmarks:
        return None, None, None, None

    results, processed_img = best_result
    h, w, _ = processed_img.shape
    lm = results.pose_landmarks.landmark

    landmarks = {
        "Nose": 0, "Left Eye": 1, "Right Eye": 2, "Left Ear": 7, "Right Ear": 8,
        "Left Shoulder": 11, "Right Shoulder": 12, "Left Elbow": 13, "Right Elbow": 14,
        "Left Wrist": 15, "Right Wrist": 16, "Left Pinky": 17, "Right Pinky": 18,
        "Left Index": 19, "Right Index": 20, "Left Thumb": 21, "Right Thumb": 22,
        "Left Hip": 23, "Right Hip": 24, "Left Knee": 25, "Right Knee": 26,
        "Left Ankle": 27, "Right Ankle": 28, "Left Heel": 29, "Right Heel": 30,
        "Left Foot Index": 31, "Right Foot Index": 32
    }

    keypoints = {}
    visibilities = {}
    for name, idx in landmarks.items():
        if idx < len(lm):
            keypoints[name] = (lm[idx].x * w, lm[idx].y * h)
            visibilities[name] = lm[idx].visibility
        else:
            keypoints[name] = (w/2, h/2)
            visibilities[name] = 0.0

    features = {}

    # Joint angles
    joint_angles = {
        'Left Elbow': ('Left Shoulder', 'Left Elbow', 'Left Wrist'),
        'Right Elbow': ('Right Shoulder', 'Right Elbow', 'Right Wrist'),
        'Left Knee': ('Left Hip', 'Left Knee', 'Left Ankle'),
        'Right Knee': ('Right Hip', 'Right Knee', 'Right Ankle'),
        'Left Hip': ('Left Shoulder', 'Left Hip', 'Left Knee'),
        'Right Hip': ('Right Shoulder', 'Right Hip', 'Right Knee'),
        'Left Shoulder': ('Left Elbow', 'Left Shoulder', 'Left Hip'),
        'Right Shoulder': ('Right Elbow', 'Right Shoulder', 'Right Hip'),
        'Left Ankle': ('Left Knee', 'Left Ankle', 'Left Foot Index'),
        'Right Ankle': ('Right Knee', 'Right Ankle', 'Right Foot Index'),
        'Torso': ('Left Shoulder', 'Left Hip', 'Right Hip'),
        'Head Tilt': ('Nose', 'Left Ear', 'Right Ear')
    }
    for angle_name, (p1, p2, p3) in joint_angles.items():
        if all(pt in keypoints for pt in [p1, p2, p3]):
            features[f'{angle_name}_Angle'] = calculate_angle(keypoints[p1], keypoints[p2], keypoints[p3])

    # Body proportions
    if all(pt in keypoints for pt in ['Left Shoulder', 'Right Shoulder', 'Left Hip', 'Right Hip']):
        shoulder_width = euclidean(keypoints['Left Shoulder'], keypoints['Right Shoulder'])
        hip_width = euclidean(keypoints['Left Hip'], keypoints['Right Hip'])
        features['Shoulder_Hip_Ratio'] = shoulder_width / (hip_width + 1e-8)
        torso_center_top = ((keypoints['Left Shoulder'][0] + keypoints['Right Shoulder'][0])/2,
                           (keypoints['Left Shoulder'][1] + keypoints['Right Shoulder'][1])/2)
        torso_center_bottom = ((keypoints['Left Hip'][0] + keypoints['Right Hip'][0])/2,
                             (keypoints['Left Hip'][1] + keypoints['Right Hip'][1])/2)
        features['Torso_Length'] = euclidean(torso_center_top, torso_center_bottom) / w

    # Limb lengths
    limb_pairs = [('Upper_Arm_L', 'Left Shoulder', 'Left Elbow'), ('Upper_Arm_R', 'Right Shoulder', 'Right Elbow'),
                  ('Forearm_L', 'Left Elbow', 'Left Wrist'), ('Forearm_R', 'Right Elbow', 'Right Wrist'),
                  ('Thigh_L', 'Left Hip', 'Left Knee'), ('Thigh_R', 'Right Hip', 'Right Knee'),
                  ('Shin_L', 'Left Knee', 'Left Ankle'), ('Shin_R', 'Right Knee', 'Right Ankle')]
    for limb_name, p1, p2 in limb_pairs:
        if p1 in keypoints and p2 in keypoints:
            features[f'{limb_name}_Length'] = euclidean(keypoints[p1], keypoints[p2]) / w

    # Symmetry
    if all(pt in keypoints for pt in ['Left Shoulder', 'Right Shoulder', 'Left Wrist', 'Right Wrist']):
        features['Arm_Symmetry'] = abs(calculate_angle(keypoints['Left Shoulder'], keypoints['Left Elbow'], keypoints['Left Wrist']) -
                                      calculate_angle(keypoints['Right Shoulder'], keypoints['Right Elbow'], keypoints['Right Wrist']))
    if all(pt in keypoints for pt in ['Left Hip', 'Right Hip', 'Left Knee', 'Right Knee']):
        features['Leg_Symmetry'] = abs(calculate_angle(keypoints['Left Hip'], keypoints['Left Knee'], keypoints['Left Ankle']) -
                                      calculate_angle(keypoints['Right Hip'], keypoints['Right Knee'], keypoints['Right Ankle']))

    # Center of mass
    major_points = ['Left Shoulder', 'Right Shoulder', 'Left Hip', 'Right Hip', 'Left Knee', 'Right Knee', 'Left Ankle', 'Right Ankle']
    valid_points = {k: v for k, v in keypoints.items() if k in major_points}
    if valid_points:
        center_of_mass = calculate_centroid(valid_points)
        features['CoM_X'] = center_of_mass[0] / w
        features['CoM_Y'] = center_of_mass[1] / h

    # Hand and foot positions
    if 'Left Wrist' in keypoints and 'Left Shoulder' in keypoints:
        features['Left_Hand_Height'] = (keypoints['Left Shoulder'][1] - keypoints['Left Wrist'][1]) / h
    if 'Right Wrist' in keypoints and 'Right Shoulder' in keypoints:
        features['Right_Hand_Height'] = (keypoints['Right Shoulder'][1] - keypoints['Right Wrist'][1]) / h
    if 'Left Ankle' in keypoints and 'Right Ankle' in keypoints:
        features['Foot_Separation'] = euclidean(keypoints['Left Ankle'], keypoints['Right Ankle']) / w

    # Triangle areas
    triangle_sets = [('Torso_Triangle', 'Left Shoulder', 'Right Shoulder', 'Left Hip'),
                     ('Left_Arm_Triangle', 'Left Shoulder', 'Left Elbow', 'Left Wrist'),
                     ('Right_Arm_Triangle', 'Right Shoulder', 'Right Elbow', 'Right Wrist'),
                     ('Left_Leg_Triangle', 'Left Hip', 'Left Knee', 'Left Ankle'),
                     ('Right_Leg_Triangle', 'Right Hip', 'Right Knee', 'Right Ankle')]
    for triangle_name, p1, p2, p3 in triangle_sets:
        if all(pt in keypoints for pt in [p1, p2, p3]):
            v1 = np.array(keypoints[p2]) - np.array(keypoints[p1])
            v2 = np.array(keypoints[p3]) - np.array(keypoints[p1])
            features[f'{triangle_name}_Area'] = abs(np.cross(v1, v2)) / 2 / (w * h)

    # Visibility
    features['Total_Visibility'] = sum(visibilities.values()) / len(visibilities)

    return features, keypoints, processed_img, visibilities

def calculate_angle(a, b, c):
    """Calculate angle at joint b (in degrees)."""
    try:
        a, b, c = np.array(a), np.array(b), np.array(c)
        ba = a - b
        bc = c - b
        norm_ba = np.linalg.norm(ba)
        norm_bc = np.linalg.norm(bc)
        if norm_ba < 1e-6 or norm_bc < 1e-6:
            return 90.0
        cosine_angle = np.dot(ba, bc) / (norm_ba * norm_bc)
        angle = np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))
        return angle
    except:
        return 90.0

def calculate_centroid(points):
    """Calculate centroid of given points."""
    try:
        points_array = np.array(list(points.values()))
        return np.mean(points_array, axis=0)
    except:
        return np.array([0.5, 0.5])

# ----------------------------
# 2. Prediction
# ----------------------------

# def predict_stance_intelligent(image_path, pipeline, feature_ranges, stance_stats, feature_columns):
#     """Run full pipeline with intelligent config selection."""
#     configs = ['high_accuracy', 'balanced', 'fast']
#     best_result = None
#     best_confidence = 0

#     for config in configs:
#         features, keypoints, processed_image, visibilities = extract_comprehensive_features(
#             cv2.imread(image_path), config=config)
#         if features is None or keypoints is None or processed_image is None or visibilities is None:
#             continue

#         if features['Total_Visibility'] < 0.6:
#             continue

#         # Convert to feature vector
#         feature_vector = [features.get(col, 0.0) for col in feature_columns]
#         probabilities = pipeline.predict_proba([feature_vector])[0]
#         stance_id = np.argmax(probabilities)
#         confidence = probabilities[stance_id]
#         stance_name = STANCE_NAMES[stance_id]

#         # Consistency score
#         consistency_score = calculate_feature_consistency(features, stance_stats, stance_name)
#         combined_confidence = (confidence * 0.7 + consistency_score * 0.3)

#         if combined_confidence > best_confidence:
#             best_confidence = combined_confidence
#             best_result = {
#                 'stance_name': stance_name,
#                 'stance_id': stance_id,
#                 'confidence': combined_confidence,
#                 'probabilities': probabilities,
#                 'features': features,
#                 'keypoints': keypoints,
#                 'processed_image': processed_image,
#                 'visibilities': visibilities,
#                 'config_used': config
#             }

#     if best_result is None:
#         return None

#     # feedback = generate_detailed_feedback(best_result['features'], feature_ranges, stance_stats, best_result['stance_name'])
#     feedback = generate_detailed_feedback(
#     best_result['features'],
#     feature_ranges,
#     stance_stats,
#     best_result['stance_name'],
#     best_result['visibilities']  # 👈 NEW
# )

#     return best_result, feedback

def predict_stance_intelligent(image_path, pipeline, feature_ranges, stance_stats, feature_columns):
    """Run full pipeline with intelligent config selection and OOD Guardrails."""
    configs = ['high_accuracy', 'balanced', 'fast']
    best_result = None
    best_confidence = 0

    # --- GUARDRAIL THRESHOLDS ---
    MIN_CONFIDENCE_THRESHOLD = 0.70  # Minimum probability to accept a prediction
    MIN_MARGIN_THRESHOLD = 0.15      # Diff between Top 1 and Top 2 prediction (prevents confusion)
    MIN_CONSISTENCY_THRESHOLD = 0.4  # If features deviate too much from statistical mean

    for config in configs:
        features, keypoints, processed_image, visibilities = extract_comprehensive_features(
            cv2.imread(image_path), config=config)
        
        if features is None or keypoints is None or processed_image is None or visibilities is None:
            continue

        if features['Total_Visibility'] < 0.6:
            continue

        # Convert to feature vector
        feature_vector = [features.get(col, 0.0) for col in feature_columns]
        
        # Get Probabilities
        probabilities = pipeline.predict_proba([feature_vector])[0]
        
        # --- 1. ENTROPY/MARGIN CHECK (Detects Karate vs Taekwondo confusion) ---
        # Sort probabilities to see the top contenders
        sorted_indices = np.argsort(probabilities)[::-1]
        top_1_idx = sorted_indices[0]
        top_2_idx = sorted_indices[1]
        
        top_1_prob = probabilities[top_1_idx]
        top_2_prob = probabilities[top_2_idx]
        
        # If the gap between the winner and runner-up is too small, the model is guessing.
        # This often happens when a Taekwondo model looks at a Karate stance.
        margin = top_1_prob - top_2_prob
        
        stance_id = top_1_idx
        stance_name = STANCE_NAMES[stance_id]

        # Consistency score (Z-Score check you already wrote - very useful here!)
        consistency_score = calculate_feature_consistency(features, stance_stats, stance_name)
        
        # Calculate a Weighted Confidence
        combined_confidence = (top_1_prob * 0.6 + consistency_score * 0.4)

        # --- 2. HARD REJECTION LOGIC ---
        # If the model isn't sure, or the physics don't match the stats, reject it.
        is_valid_prediction = (
            top_1_prob >= MIN_CONFIDENCE_THRESHOLD and 
            margin >= MIN_MARGIN_THRESHOLD and
            consistency_score >= MIN_CONSISTENCY_THRESHOLD
        )

        if is_valid_prediction and combined_confidence > best_confidence:
            best_confidence = combined_confidence
            best_result = {
                'stance_name': stance_name,
                'stance_id': stance_id,
                'confidence': combined_confidence,
                'probabilities': probabilities,
                'features': features,
                'keypoints': keypoints,
                'processed_image': processed_image,
                'visibilities': visibilities,
                'config_used': config
            }

    # If no config produced a result that passed our thresholds:
    if best_result is None:
        # RETURN AN "UNKNOWN" RESULT INSTANCE
        return {
            'stance_name': "Unknown / Invalid Stance",
            'confidence': 0.0,
            'is_unknown': True  # Flag for your API to handle 400 or custom message
        }, ["Pose does not match trained Taekwondo criteria."]

    # Generate feedback only for valid results
    feedback = generate_detailed_feedback(
        best_result['features'],
        feature_ranges,
        stance_stats,
        best_result['stance_name'],
        best_result['visibilities']
    )

    return best_result, feedback

def calculate_feature_consistency(features, stance_stats, stance_name):
    """Calculate consistency score based on z-score."""
    if stance_name not in stance_stats:
        return 0.5
    consistency_scores = []
    stats = stance_stats[stance_name]
    for feature_name, value in features.items():
        if feature_name in stats and stats[feature_name]['std'] > 0:
            z_score = abs((value - stats[feature_name]['mean']) / stats[feature_name]['std'])
            consistency = max(0, 1 - (z_score / 3))
            consistency_scores.append(consistency)
    return np.mean(consistency_scores) if consistency_scores else 0.5

# ----------------------------
# 3. Feedback
# ----------------------------

# def generate_detailed_feedback(features, feature_ranges, stance_stats, stance_name):
#     """Generate comprehensive feedback with emoji indicators."""
#     if stance_name not in feature_ranges:
#         return ["Unable to provide specific feedback for this stance."]

#     feedback = []
#     ranges = feature_ranges[stance_name]
#     stats = stance_stats[stance_name]

#     # Key angles feedback
#     key_angles = [col for col in features.keys() if 'Angle' in col]
#     for angle_name in key_angles:
#         if angle_name in ranges and angle_name in features:
#             actual = features[angle_name]
#             low, high = ranges[angle_name]
#             ideal = stats[angle_name]['median']
#             deviation = abs(actual - ideal)
#             if actual < low:
#                 feedback.append(f"🔴 {angle_name.replace('_', ' ')}: {actual:.1f}° (too small, ideal: {ideal:.1f}°)")
#             elif actual > high:
#                 feedback.append(f"🔴 {angle_name.replace('_', ' ')}: {actual:.1f}° (too large, ideal: {ideal:.1f}°)")
#             else:
#                 feedback.append(f"✅ {angle_name.replace('_', ' ')}: {actual:.1f}° (good)")

#     # Symmetry feedback
#     if 'Arm_Symmetry' in features:
#         arm_sym = features['Arm_Symmetry']
#         if arm_sym > 20:
#             feedback.append(f"🔴 Arm symmetry: {arm_sym:.1f}° difference (try to balance both arms)")
#         elif arm_sym > 10:
#             feedback.append(f"🟡 Arm symmetry: {arm_sym:.1f}° difference (minor imbalance)")
#         else:
#             feedback.append(f"✅ Arm symmetry: {arm_sym:.1f}° difference (well balanced)")

#     if 'Leg_Symmetry' in features:
#         leg_sym = features['Leg_Symmetry']
#         if leg_sym > 20:
#             feedback.append(f"🔴 Leg symmetry: {leg_sym:.1f}° difference (try to balance both legs)")
#         elif leg_sym > 10:
#             feedback.append(f"🟡 Leg symmetry: {leg_sym:.1f}° difference (minor imbalance)")
#         else:
#             feedback.append(f"✅ Leg symmetry: {leg_sym:.1f}° difference (well balanced)")

#     # Stability feedback
#     if 'Foot_Separation' in features:
#         foot_sep = features['Foot_Separation']
#         if foot_sep < 0.1:
#             feedback.append("🟡 Foot separation seems narrow - consider widening stance")
#         elif foot_sep > 0.4:
#             feedback.append("🟡 Foot separation seems wide - consider narrowing stance")

#     return feedback[:8]

# def generate_detailed_feedback(features, feature_ranges, stance_stats, stance_name, visibilities):
#     if stance_name not in feature_ranges:
#         return ["Unable to provide specific feedback for this stance."]

#     feedback = []
#     ranges = feature_ranges[stance_name]
#     stats = stance_stats[stance_name]

#     # 🔍 Helper to check visibility of required keypoints
#     def is_visible(*points, threshold=0.6):
#         return all(visibilities.get(p, 0) > threshold for p in points)

#     # 1️⃣ Angle feedback
#     key_angles = [col for col in features.keys() if 'Angle' in col]
#     for angle_name in key_angles:
#         # Joint name format: "Left Elbow Angle" → "Left Elbow"
#         joint = angle_name.replace("_Angle", "").strip()

#         # Only check if this joint is visible
#         if not is_visible(joint):
#             continue

#         if angle_name in ranges:
#             actual = features[angle_name]
#             low, high = ranges[angle_name]
#             ideal = stats[angle_name]['median']

#             if actual < low:
#                 feedback.append(f"🔴 {joint}: {actual:.1f}° (too small, ideally {ideal:.1f}°)")
#             elif actual > high:
#                 feedback.append(f"🔴 {joint}: {actual:.1f}° (too large, ideally {ideal:.1f}°)")
#             else:
#                 feedback.append(f"✅ {joint}: {actual:.1f}° (Good)")

#     # 2️⃣ Arm symmetry — only if BOTH arms are visible
#     if is_visible('Left Wrist', 'Right Wrist', 'Left Shoulder', 'Right Shoulder'):
#         arm_sym = features.get('Arm_Symmetry', None)
#         if arm_sym is not None:
#             if arm_sym > 20:
#                 feedback.append(f"🔴 Arm symmetry: {arm_sym:.1f}° difference (too uneven)")
#             elif arm_sym > 10:
#                 feedback.append(f"🟡 Arm symmetry: {arm_sym:.1f}° difference (could improve)")
#             else:
#                 feedback.append(f"✅ Arm symmetry: {arm_sym:.1f}° (excellent)")

#     # 3️⃣ Leg symmetry — only if BOTH legs are visible
#     if is_visible('Left Ankle', 'Right Ankle', 'Left Hip', 'Right Hip'):
#         leg_sym = features.get('Leg_Symmetry', None)
#         if leg_sym is not None:
#             if leg_sym > 20:
#                 feedback.append(f"🔴 Leg symmetry: {leg_sym:.1f}° difference (too uneven)")
#             elif leg_sym > 10:
#                 feedback.append(f"🟡 Leg symmetry: {leg_sym:.1f}° difference (could improve)")
#             else:
#                 feedback.append(f"✅ Leg symmetry: {leg_sym:.1f}° (excellent)")

#     # 4️⃣ Foot separation — only if both ankles visible
#     if is_visible('Left Ankle', 'Right Ankle'):
#         if 'Foot_Separation' in features:
#             foot_sep = features['Foot_Separation']
#             if foot_sep < 0.1:
#                 feedback.append("🟡 Foot separation looks narrow")
#             elif foot_sep > 0.4:
#                 feedback.append("🟡 Foot separation seems wide")
#             else:
#                 feedback.append("✅ Good foot distance")

#     # 5️⃣ Torso tilt — only if torso is visible
#     if is_visible('Left Shoulder', 'Right Shoulder'):
#         if 'Torso_Lean' in features:
#             lean = features['Torso_Lean']
#             if abs(lean) > 10:
#                 feedback.append(f"🟡 Torso tilt: {lean:.1f}° (try keeping more upright)")
#             else:
#                 feedback.append("✅ Torso posture looks stable")

#     return feedback[:8]  # 🔥 Limit to max 8 feedbacks


def generate_detailed_feedback(features, feature_ranges, stance_stats, stance_name, visibilities):
    if stance_name not in feature_ranges:
        return ["Unable to provide specific feedback for this stance."]

    feedback = []
    ranges = feature_ranges[stance_name]
    stats = stance_stats[stance_name]

    # Helper: Check if joints are visible
    def is_visible(*points, threshold=0.6):
        return all(visibilities.get(p, 0) > threshold for p in points)

    # 🔍 Convert degree error into human language
    def describe_adjustment(actual, ideal):
        diff = actual - ideal
        abs_diff = abs(diff)

        if abs_diff < 5:
            return "looks good"
        elif abs_diff < 10:
            return "adjust slightly"
        elif abs_diff < 20:
            return "adjust noticeably"
        else:
            return "make a strong correction"

    # Suggest movement direction
    def movement_direction(joint, actual, ideal):
        diff = actual - ideal
        if "Elbow" in joint or "Shoulder" in joint:
            if diff > 0:
                return f"lower your {joint.lower()}"
            else:
                return f"raise your {joint.lower()}"
        if "Knee" in joint or "Hip" in joint or "Ankle" in joint:
            if diff > 0:
                return f"straighten your {joint.lower()}"
            else:
                return f"bend your {joint.lower()}"
        if "Head" in joint:
            if diff > 0:
                return f"tilt your head slightly left"
            else:
                return f"tilt your head slightly right"
        return f"adjust your {joint.lower()}"

    # 1️⃣ Angle Feedback
    key_angles = [col for col in features.keys() if 'Angle' in col]
    for angle_name in key_angles:
        joint = angle_name.replace("_Angle", "").strip()

        if not is_visible(joint):  # Skip hidden joints
            continue

        if angle_name in ranges:
            actual = features[angle_name]
            low, high = ranges[angle_name]
            ideal = stats[angle_name]['median']

            if actual < low or actual > high:
                adj = describe_adjustment(actual, ideal)
                move = movement_direction(joint, actual, ideal)
                feedback.append(f"🔴 {joint}: {move}, {adj}")
            else:
                feedback.append(f"✅ {joint}: looks good")

    # 2️⃣ Arm Symmetry
    if is_visible('Left Wrist', 'Right Wrist', 'Left Shoulder', 'Right Shoulder'):
        arm_sym = features.get('Arm_Symmetry')
        if arm_sym is not None:
            if arm_sym > 20:
                feedback.append("🔴 Try balancing both arms equally")
            elif arm_sym > 10:
                feedback.append("🟡 Slight imbalance between arms")
            else:
                feedback.append("✅ Arm balance is good")

    # 3️⃣ Leg Symmetry
    if is_visible('Left Ankle', 'Right Ankle', 'Left Hip', 'Right Hip'):
        leg_sym = features.get('Leg_Symmetry')
        if leg_sym is not None:
            if leg_sym > 20:
                feedback.append("🔴 Legs appear uneven – try balancing them")
            elif leg_sym > 10:
                feedback.append("🟡 Minor imbalance in leg position")
            else:
                feedback.append("✅ Leg positioning is well balanced")

    # 4️⃣ Foot Separation
    if is_visible('Left Ankle', 'Right Ankle'):
        foot_sep = features.get('Foot_Separation')
        if foot_sep is not None:
            if foot_sep < 0.1:
                feedback.append("🟡 Try widening your feet slightly")
            elif foot_sep > 0.4:
                feedback.append("🟡 Feet appear too wide – bring them slightly closer")
            else:
                feedback.append("✅ Good foot distance")

    # 5️⃣ Torso Alignment
    if is_visible('Left Shoulder', 'Right Shoulder'):
        lean = features.get('Torso_Lean', 0)
        if abs(lean) > 10:
            feedback.append("🟡 Try keeping your torso more upright")
        else:
            feedback.append("✅ Good torso posture")

    return feedback[:8]  # Max 8 feedback suggestions



# ----------------------------
# 4. Visualization (for API response preparation)
# ----------------------------

def visualize_pose_advanced(image, keypoints, visibilities=None):
    """Generate pose visualization image."""
    if image is None or keypoints is None:
        return None

    vis_image = image.copy()

    def get_color(visibility):
        if visibility > 0.8: return (0, 255, 0)  # Green
        elif visibility > 0.6: return (0, 255, 255)  # Yellow
        else: return (0, 0, 255)  # Red

    # Draw keypoints
    for name, (x, y) in keypoints.items():
        visibility = visibilities.get(name, 0.8) if visibilities else 0.8
        color = get_color(visibility)
        cv2.circle(vis_image, (int(x), int(y)), 6, color, -1)
        cv2.circle(vis_image, (int(x), int(y)), 8, (255, 255, 255), 2)

    # Draw connections
    connections = [
        ("Left Shoulder", "Left Elbow"), ("Left Elbow", "Left Wrist"),
        ("Right Shoulder", "Right Elbow"), ("Right Elbow", "Right Wrist"),
        ("Left Shoulder", "Right Shoulder"), ("Left Shoulder", "Left Hip"),
        ("Right Shoulder", "Right Hip"), ("Left Hip", "Right Hip"),
        ("Left Hip", "Left Knee"), ("Left Knee", "Left Ankle"),
        ("Right Hip", "Right Knee"), ("Right Knee", "Right Ankle"),
    ]
    for start, end in connections:
        if start in keypoints and end in keypoints:
            start_vis = visibilities.get(start, 0.8) if visibilities else 0.8
            end_vis = visibilities.get(end, 0.8) if visibilities else 0.8
            avg_vis = (start_vis + end_vis) / 2
            thickness = int(2 + avg_vis * 3)
            color = get_color(avg_vis)
            start_pt = (int(keypoints[start][0]), int(keypoints[start][1]))
            end_pt = (int(keypoints[end][0]), int(keypoints[end][1]))
            cv2.line(vis_image, start_pt, end_pt, color, thickness)

    return vis_image