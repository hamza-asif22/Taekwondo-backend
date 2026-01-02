from fastapi import FastAPI
from app.api import routes_pose

app = FastAPI(
    title="Taekwondo Pose Analysis API",
    description="API for detecting and analyzing Taekwondo stances using Mediapipe + ML",
    version="1.0.0"
)

# Register routes
app.include_router(routes_pose.router)

@app.get("/")
def root():
    return {"message": "🥋 Taekwondo Pose API is running!"}
