import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

STANCE_NAMES = ['fighting-stance', 'front-kick', 'high-block', 'low-block', 'middle-block', 'poomsae-stance', 'squat']
# MODEL_PATH = "D:/Final Year Project/Development/kokoro_dojo/model_api/app/models/taekwondo_model.pkl"
MODEL_PATH = "D:/Final Year Project/Development/ai_model/model_api/app/models/taekwondo_pose_model.pkl"
STATS_PATH = "D:/Final Year Project/Development/ai_model/model_api/app/models/pose_statistics.pkl"