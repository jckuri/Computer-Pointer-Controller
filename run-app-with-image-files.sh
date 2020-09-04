FACE_DETECTION_MODEL=models/intel/face-detection-adas-binary-0001/FP32-INT1/face-detection-adas-binary-0001.xml
FACIAL_LANDMARK_MODEL=models/intel/landmarks-regression-retail-0009/FP32/landmarks-regression-retail-0009.xml
HEAD_POSE_MODEL=models/intel/head-pose-estimation-adas-0001/FP32/head-pose-estimation-adas-0001.xml
GAZE_ESTIMATION_MODEL=models/intel/gaze-estimation-adas-0002/FP32/gaze-estimation-adas-0002.xml

python src/eye_pointer_app.py --input_type image --input_file bin/demo.mp4.20.png --output_file bin/output.demo.mp4.20.png --device CPU --face_detection_model $FACE_DETECTION_MODEL --facial_landmark_model $FACIAL_LANDMARK_MODEL --head_pose_model $HEAD_POSE_MODEL --gaze_estimation_model $GAZE_ESTIMATION_MODEL --show_face_detection --show_facial_landmarks --show_head_pose --show_gaze_estimation

python src/eye_pointer_app.py --input_type image --input_file bin/demo.mp4.250.png --output_file bin/output.demo.mp4.250.png --device CPU --face_detection_model $FACE_DETECTION_MODEL --facial_landmark_model $FACIAL_LANDMARK_MODEL --head_pose_model $HEAD_POSE_MODEL --gaze_estimation_model $GAZE_ESTIMATION_MODEL --show_face_detection --show_facial_landmarks --show_head_pose --show_gaze_estimation
