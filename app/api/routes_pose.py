# from fastapi import APIRouter, UploadFile, File
# from fastapi.responses import JSONResponse, FileResponse
# import tempfile, os, cv2

# from app.services.analysis import predict_stance_intelligent
# from app.services.model_utils import load_model_and_statistics
# from app.services.visualize import visualize_pose_advanced

# router = APIRouter(prefix="/pose", tags=["pose"])

# # Load model and statistics once at startup
# pipeline, feature_ranges, stance_stats, feature_columns = load_model_and_statistics()

# @router.post("/analyze")
# async def analyze_pose(file: UploadFile = File(...)):
#     """
#     Analyze an uploaded pose image and return stance + feedback.
#     """
#     try:
#         tmp = tempfile.mktemp(suffix=".jpg")
#         with open(tmp, "wb") as f:
#             f.write(file.file.read())

#         result = predict_stance_intelligent(tmp, pipeline, feature_ranges, stance_stats, feature_columns)
#         if not result:
#             return JSONResponse({"error": "Pose not detected"}, status_code=400)

#         stance, feedback = result
#         return {
#             "stance": stance["stance_name"],
#             "confidence": stance["confidence"],
#             "feedback": feedback,
#             "metrics": {
#                 "visibility": stance["features"].get("Total_Visibility", 0),
#                 "arm_symmetry": stance["features"].get("Arm_Symmetry", 0),
#                 "leg_symmetry": stance["features"].get("Leg_Symmetry", 0),
#                 "foot_separation": stance["features"].get("Foot_Separation", 0),
#             }
#         }
#     except Exception as e:
#         return JSONResponse({"error": str(e)}, status_code=500)

# @router.post("/analyze_with_image")
# async def analyze_with_image(file: UploadFile = File(...)):
#     tmp = tempfile.mktemp(suffix=".jpg")
#     with open(tmp, "wb") as f:
#         f.write(file.file.read())

#     result, feedback = predict_stance_intelligent(tmp, pipeline, feature_ranges, stance_stats, feature_columns)
#     if not result:
#         return JSONResponse({"error": "Pose not detected"}, status_code=400)

#     out_path = os.path.join(tempfile.gettempdir(), "pose_result.jpg")
#     vis_img = visualize_pose_advanced(result["processed_image"], result["keypoints"], result["visibilities"])
#     cv2.imwrite(out_path, vis_img)

#     return {
#         "analysis": {
#             "stance": result["stance_name"],
#             "confidence": result["confidence"],
#             "probabilities": result["probabilities"].tolist(),
#             "config_used": result["config_used"],
#             "feedback": feedback,
#         },
#         "image_url": f"/pose/download/pose_result.jpg"
#     }



from fastapi import APIRouter, UploadFile, File
from fastapi.responses import JSONResponse
import tempfile, os, cv2

# Import your updated function
from app.services.analysis import predict_stance_intelligent
from app.services.model_utils import load_model_and_statistics
from app.services.visualize import visualize_pose_advanced

router = APIRouter(prefix="/pose", tags=["pose"])

# Load model and statistics once at startup
pipeline, feature_ranges, stance_stats, feature_columns = load_model_and_statistics()

@router.post("/analyze")
async def analyze_pose(file: UploadFile = File(...)):
    """
    Analyze an uploaded pose image and return stance + feedback.
    """
    try:
        # Save temp file
        tmp = tempfile.mktemp(suffix=".jpg")
        with open(tmp, "wb") as f:
            f.write(file.file.read())

        # 1. CALL THE INTELLIGENT FUNCTION
        # It always returns a tuple: (best_result_dict, feedback_list)
        best_result, feedback = predict_stance_intelligent(
            tmp, pipeline, feature_ranges, stance_stats, feature_columns
        )

        # 2. CHECK FOR "UNKNOWN" / GUARDRAIL REJECTION
        # This prevents the Karate vs Taekwondo confusion
        if best_result.get("is_unknown"):
            return JSONResponse(
                status_code=400,
                content={
                    "error": "Stance not recognized",
                    "message": "The pose does not match valid Taekwondo criteria (or is a different martial art).",
                    "confidence": 0.0,
                    "feedback": feedback
                }
            )

        # 3. IF VALID, EXTRACT METRICS SAFELY
        # We perform .get() safely because we know 'features' exists now
        metrics = {
            "visibility": best_result["features"].get("Total_Visibility", 0),
            "arm_symmetry": best_result["features"].get("Arm_Symmetry", 0),
            "leg_symmetry": best_result["features"].get("Leg_Symmetry", 0),
            "foot_separation": best_result["features"].get("Foot_Separation", 0),
        }

        return {
            "stance": best_result["stance_name"],
            "confidence": best_result["confidence"],
            "feedback": feedback,
            "metrics": metrics
        }

    except Exception as e:
        import traceback
        traceback.print_exc() # Print error to server console for debugging
        return JSONResponse({"error": str(e)}, status_code=500)


@router.post("/analyze_with_image")
async def analyze_with_image(file: UploadFile = File(...)):
    """
    Analyze pose AND return a processed image with skeleton overlay.
    """
    try:
        tmp = tempfile.mktemp(suffix=".jpg")
        with open(tmp, "wb") as f:
            f.write(file.file.read())

        best_result, feedback = predict_stance_intelligent(
            tmp, pipeline, feature_ranges, stance_stats, feature_columns
        )

        # 1. CHECK FOR UNKNOWN BEFORE VISUALIZING
        if best_result.get("is_unknown"):
            return JSONResponse(
                status_code=400,
                content={
                    "error": "Stance not recognized",
                    "message": "Pose rejected by validation guardrails.",
                    "confidence": 0.0,
                    "feedback": feedback
                }
            )

        # 2. GENERATE VISUALIZATION (Only if valid)
        out_path = os.path.join(tempfile.gettempdir(), "pose_result.jpg")
        
        # We can safely access keypoints/visibilities because we passed the check above
        vis_img = visualize_pose_advanced(
            best_result["processed_image"], 
            best_result["keypoints"], 
            best_result["visibilities"]
        )
        
        if vis_img is not None:
            cv2.imwrite(out_path, vis_img)
        else:
            # Fallback if visualization failed for some reason
            return JSONResponse({"error": "Could not generate visualization"}, status_code=500)

        return {
            "analysis": {
                "stance": best_result["stance_name"],
                "confidence": best_result["confidence"],
                "probabilities": best_result["probabilities"].tolist(),
                "config_used": best_result["config_used"],
                "feedback": feedback,
            },
            "image_url": f"/pose/download/pose_result.jpg" 
            # Ensure you have a route that serves this file!
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=500)