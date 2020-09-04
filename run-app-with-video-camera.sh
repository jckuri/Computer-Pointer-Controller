FACE_DETECTION_MODEL=models/intel/face-detection-adas-binary-0001/FP32-INT1/face-detection-adas-binary-0001.xml
FACIAL_LANDMARK_MODEL=models/intel/landmarks-regression-retail-0009/FP16/landmarks-regression-retail-0009.xml
HEAD_POSE_MODEL=models/intel/head-pose-estimation-adas-0001/FP16/head-pose-estimation-adas-0001.xml
GAZE_ESTIMATION_MODEL=models/intel/gaze-estimation-adas-0002/FP16/gaze-estimation-adas-0002.xml

python src/eye_pointer_app.py --input_type cam --output_file bin/output_video_camera.mp4 --device CPU --face_detection_model $FACE_DETECTION_MODEL --facial_landmark_model $FACIAL_LANDMARK_MODEL --head_pose_model $HEAD_POSE_MODEL --gaze_estimation_model $GAZE_ESTIMATION_MODEL --show_face_detection --show_facial_landmarks --show_head_pose --show_gaze_estimation --show_pointer

