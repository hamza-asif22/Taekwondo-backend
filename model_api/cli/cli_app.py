import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
from app.services.analysis import predict_stance_intelligent
from app.services.model_utils import load_model_and_statistics
from app.services.visualize import visualize_pose_advanced

# Load ML model & stats once
pipeline, feature_ranges, stance_stats, feature_columns = load_model_and_statistics()

# --------------------------
# CLI / Tkinter Application
# --------------------------

class TaekwondoPoseApp:
    def __init__(self, root):
        self.root = root
        self.root.title("🥋 Taekwondo Pose Analyzer")
        self.root.geometry("400x250")

        tk.Label(root, text="Taekwondo Pose Analyzer", font=("Arial", 16, "bold")).pack(pady=10)

        self.upload_btn = tk.Button(root, text="Upload Image", command=self.upload_image)
        self.upload_btn.pack(pady=10)

        self.result_label = tk.Label(root, text="", font=("Arial", 12), wraplength=350, justify="center")
        self.result_label.pack(pady=10)

    def upload_image(self):
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image Files", "*.jpg *.jpeg *.png")]
        )
        if not file_path:
            return

        try:
            result = predict_stance_intelligent(file_path, pipeline, feature_ranges, stance_stats, feature_columns)

            if not result:
                messagebox.showerror("Error", "Pose could not be detected.")
                return

            stance, feedback = result
            stance_name = stance["stance_name"]
            confidence = stance["confidence"]

            # Show result in GUI
            self.result_label.config(
                text=f"Stance: {stance_name}\nConfidence: {confidence:.2f}\n\nFeedback:\n- " + "\n- ".join(feedback)
            )

            # Show visualization in OpenCV window
            vis_img = visualize_pose_advanced(
                stance["processed_image"], stance["keypoints"],
                stance_name, confidence, stance["visibilities"]
            )
            cv2.imshow("Pose Analysis", vis_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        except Exception as e:
            messagebox.showerror("Error", str(e))

# --------------------------
# Run Application
# --------------------------

if __name__ == "__main__":
    root = tk.Tk()
    app = TaekwondoPoseApp(root)
    root.mainloop()
