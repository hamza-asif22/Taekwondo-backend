import cv2

def visualize_pose_advanced(image, keypoints, stance_name="", confidence=0.0, visibilities=None):
    """Draw pose landmarks with stance label."""
    if image is None or keypoints is None:
        return image

    vis_image = image.copy()

    def get_color(visibility):
        if visibility > 0.8: return (0, 255, 0)
        elif visibility > 0.6: return (0, 255, 255)
        else: return (0, 0, 255)

    # Draw keypoints
    for name, (x, y) in keypoints.items():
        visibility = visibilities.get(name, 0.8) if visibilities else 0.8
        color = get_color(visibility)
        cv2.circle(vis_image, (int(x), int(y)), 6, color, -1)
        cv2.circle(vis_image, (int(x), int(y)), 8, (255, 255, 255), 2)

    # Draw stance name
    label = f"{stance_name} ({confidence:.1%})"
    cv2.putText(vis_image, label, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

    return vis_image
