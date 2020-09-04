from argparse import ArgumentParser
import models
import input_feeder
import mouse_controller
import cv2
import numpy as np
import logging
import time

def seconds_to_minutes_string(seconds):
    int_seconds = int(seconds)
    m = int_seconds // 60
    s = int_seconds % 60
    if m == 0: return '{} seconds'.format(s)
    return '{} minutes {} seconds'.format(m, s)

class Main:

    def __init__(self):
        TIME0 = time.time()
        self.args = self.build_argparser().parse_args()
        #logging.basicConfig(filename = self.args.log_file, filemode='w', level=logging.DEBUG, format='%(asctime)s %(levelname)s: %(message)s')
        logging.basicConfig(handlers=[logging.FileHandler(self.args.log_file), logging.StreamHandler()], level=logging.DEBUG, format='%(asctime)s %(levelname)s: %(message)s')
        self.load_models()
        self.process_input_feeder()       
        self.TOTAL_TIME = time.time() - TIME0
        if self.args.show_time_stats: self.show_time_stats() 

    def build_argparser(self):
        parser = ArgumentParser()
        parser.add_argument('--input_type', required = True, type = str, help = 'Input type can be "video", "cam", or "image".')
        parser.add_argument('--input_file', type = str, default = '', help = 'Path to image or video file.')
        parser.add_argument('--output_file', required = True, type = str, help = 'Path to the output video file.')
        parser.add_argument('--log_file', type = str, default = 'computer_pointer_controller.log', help = 'Path to the log file.')
        parser.add_argument('--device', type = str, default = 'CPU', help = 'Specify the target device to infer on: CPU (default), GPU, FPGA, or MYRIAD are acceptable.')
        parser.add_argument('--extensions', type = str, default=None, help = 'Specify CPU extensions.')
        parser.add_argument('--prob_threshold', type=float, default = 0.5, help = 'Probability threshold for filtering detections. (0.5 by default)')
        parser.add_argument('--face_detection_model', required = True, type = str, help = 'Path to the XML file with the face detection model.')
        parser.add_argument('--facial_landmark_model', required = True, type = str, help = 'Path to the XML file with the facial landmark model.')
        parser.add_argument('--head_pose_model', required = True, type = str, help = 'Path to the XML file with the head pose model.')
        parser.add_argument('--gaze_estimation_model', required = True, type = str, help = 'Path to the XML file with the gaze estimation model.')
        parser.add_argument('--show_face_detection', action='store_true', help = 'Show face detection in output video.')
        parser.add_argument('--show_facial_landmarks', action='store_true', help = 'Show facial landmarks in output video.')
        parser.add_argument('--show_head_pose', action='store_true', help = 'Show head pose in output video.')
        parser.add_argument('--show_gaze_estimation', action='store_true', help = 'Show gaze estimation in output video.')
        parser.add_argument('--show_pointer', action='store_true', help = 'Show pointer in output video.')
        parser.add_argument('--verbose', action='store_true', help = 'Print all inferences of models in terminal.')
        parser.add_argument('--show_time_stats', action='store_true', help = 'Show summary of time statistics.')
        parser.add_argument('--experiment_name', type = str, default = 'Experiment X', help = 'Name of experiment.')
        return parser

    def load_models(self):
        logging.info('')
        logging.info(self.args)
        logging.info('')
        time0 = time.time()
        self.face_detection_model = models.FaceDetectionModel('Face Detection Model', self, self.args.face_detection_model)
        self.facial_landmark_model = models.FacialLandmarkModel('Facial Landmark Model', self, self.args.facial_landmark_model)
        self.head_pose_model = models.HeadPoseModel('Head Pose Model', self, self.args.head_pose_model)
        self.gaze_estimation_model = models.GazeEstimationModel('Gaze Estimation Model', self, self.args.gaze_estimation_model)
        self.face_detection_model.load_model()
        self.facial_landmark_model.load_model()
        self.head_pose_model.load_model()
        self.gaze_estimation_model.load_model()
        self.total_load_time = time.time() - time0

    def process_input_feeder(self):
        self.input_feeder = input_feeder.InputFeeder(self, input_type = self.args.input_type, input_file = self.args.input_file)
        self.input_feeder.load_data()
        self.mouse = mouse_controller.MouseController(self, precision = 40, duration_seconds = 0)
        if self.args.input_type == 'image':
            for ret, frame in self.input_feeder.next_batch():
                output_frame = self.process_frame(frame)
                cv2.imwrite(self.args.output_file, output_frame)
                if not ret: break
            return
        video_writer = self.input_feeder.get_video_writer(self.args.output_file)
        frame_index = 0
        time0 = time.time()
        for ret, frame in self.input_feeder.next_batch():
            if not ret: break
            frame_index += 1
            #if frame_index % 10 == 0: cv2.imwrite('{}.{}.png'.format(self.args.input_file, frame_index), frame)
            output_frame = self.process_frame(frame)
            video_writer.write(output_frame)            
        self.total_prediction_time = time.time() - time0
        self.input_feeder.close()
        video_writer.release()

    def process_frame(self, frame):
        self.face_detection_model.image = frame
        self.face_detection_model.predict()
        if len(self.face_detection_model.fds) == 0: 
            self.gaze_estimation_model.gaze_vector = np.array([0, 0, 0])
            self.move_mouse(self.face_detection_model.output_image)
            return self.face_detection_model.output_image
        self.facial_landmark_model.fds = self.face_detection_model.fds
        self.facial_landmark_model.face_image = self.face_detection_model.face_image
        self.facial_landmark_model.image = frame
        self.facial_landmark_model.output_image = self.face_detection_model.output_image
        self.facial_landmark_model.predict()
        self.head_pose_model.image = frame
        self.head_pose_model.face_image = self.face_detection_model.face_image
        self.head_pose_model.output_image = self.facial_landmark_model.output_image
        self.head_pose_model.fds = self.face_detection_model.fds
        self.head_pose_model.predict()
        self.gaze_estimation_model.landmarks = self.facial_landmark_model.landmarks
        self.gaze_estimation_model.left_eye_image = self.facial_landmark_model.eye0
        self.gaze_estimation_model.right_eye_image = self.facial_landmark_model.eye1
        self.gaze_estimation_model.head_pose_angles = self.head_pose_model.pose.to_numpy()
        self.gaze_estimation_model.output_image = self.head_pose_model.output_image
        self.gaze_estimation_model.predict()
        self.move_mouse(self.gaze_estimation_model.output_image)
        return self.gaze_estimation_model.output_image

    def move_mouse(self, out_image):
        gaze = self.gaze_estimation_model.gaze_vector
        self.mouse.move(gaze[0], gaze[1])
        mouse_pointer = self.mouse.get_position()        
        image_size = out_image.shape
        mouse_pointer = models.round_point((mouse_pointer[0] * image_size[1] / self.mouse.screen_size[0], mouse_pointer[1] * image_size[0] / self.mouse.screen_size[1]))
        if self.args.show_pointer: cv2.circle(out_image, mouse_pointer, radius = 16, color = (0, 0, 255), thickness = 8) 
        if self.args.verbose: logging.info('mouse_pointer={}'.format(mouse_pointer))

    def show_time_stats(self):
        logging.info('\tTIME STATISTICS:\t{}'.format(self.args.experiment_name))
        models = [self.face_detection_model, self.facial_landmark_model, self.head_pose_model, self.gaze_estimation_model]
        logging.info('\tMODEL\tLOAD TIME [SECONDS]\tTOTAL INFERENCE TIME [SECONDS]\tAVG INFERENCE TIME [SECONDS]\tFRAMES PER SECOND')
        load_time_sum = 0
        inference_time_sum = 0
        for model in models:
            model.compute_time_stats()
            load_time_sum += model.load_time
            inference_time_sum += model.total_inference_time
            logging.info('\t{}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}'.format(model.model_name, model.load_time, model.total_inference_time, model.avg_inference_time, model.frames_per_second))
        logging.info('\tMODEL\tTOTAL LOAD TIME\tTOTAL INFERENCE TIME\tOTHER PROCESSES\tOVERALL TIME')
        other = self.TOTAL_TIME - load_time_sum - inference_time_sum
        logging.info('\t{}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}'.format(self.args.experiment_name, load_time_sum, inference_time_sum, other, self.TOTAL_TIME))

if __name__ == '__main__':
    Main()
