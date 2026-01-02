import joblib
import pickle
from app.core.config import MODEL_PATH, STATS_PATH

def load_model_and_statistics():
    try:
        pipeline = joblib.load(MODEL_PATH)
        with open(STATS_PATH, "rb") as f:
            stats_data = pickle.load(f)
        return (
            pipeline,
            stats_data["feature_ranges"],
            stats_data["stance_stats"],
            stats_data["feature_columns"]
        )
    except Exception as e:
        print(f"❌ Error loading model/statistics: {str(e)}")
        return None, None, None, None