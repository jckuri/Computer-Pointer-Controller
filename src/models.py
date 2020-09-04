from openvino.inference_engine import IECore
import cv2
import time
import numpy as np
import logging

supported_devices = ['CPU', 'GPU', 'FPGA', 'MYRIAD']

def get_size(image):
    width = image.shape[1]
    height = image.shape[0]
    return (width, height)

class AbstractModel:

    def __init__(self, model_name, main_app, model_xml):
        self.model_name = model_name
        self.main_app = main_app
        self.prob_threshold = main_app.args.prob_threshold
        self.model_xml = model_xml
        self.model_bin = model_xml[:-3] + 'bin'
        self.device = main_app.args.device
        self.extensions = main_app.args.extensions
        self.inference_times = []
        self.prediction_times = []

    def load_model(self):
        logging.info('Loading model {}'.format(self.model_xml))
        time0 = time.time()
        self.check_devices()
        self.core = IECore()
        self.model = self.core.read_network(self.model_xml, self.model_bin)
        if self.extensions is not None:
            self.core.add_extension(self.extensions, self.device)
        self.check_model()            
        self.neural_network = self.core.load_network(self.model, self.device)
        self.input_keys = list(self.model.inputs)
        #logging.info('input_keys: {}'.format(self.input_keys))
        self.load_time = time.time() - time0

    def check_devices(self):
        if self.device not in supported_devices:
            logging.error('Device {} is not a supported device: {}'.format(self.device))
            exit(1)

    def check_model(self):
        supported_layers = self.core.query_network(self.model, self.device)
        layers = self.model.layers.keys()
        unsupported_layers = [layer for layer in layers if layer not in supported_layers]
        if len(unsupported_layers) > 0:
            logging.error('Some layers are not supported: {}'.format(unsupported_layers))
            exit(1)

    def start_inference_time(self):
        self.inference_time0 = time.time()

    def stop_inference_time(self):
        self.inference_time = time.time() - self.inference_time0
        self.inference_times.append(self.inference_time)

    def start_prediction_time(self):
        self.prediction_time0 = time.time()

    def stop_prediction_time(self):
        self.prediction_time = time.time() - self.prediction_time0
        self.prediction_times.append(self.prediction_time)

    def compute_time_stats(self):
        self.total_inference_time = np.sum(self.inference_times)
        self.total_prediction_time = np.sum(self.prediction_times)
        self.n_frames = len(self.prediction_times)
        self.avg_inference_time = self.total_inference_time / self.n_frames
        self.frames_per_second = self.n_frames / self.total_inference_time
        self.avg_prediction_time = self.total_prediction_time / self.n_frames

    def predict(self):
        self.start_prediction_time()
        self.preprocess_input()
        self.inference()
        self.preprocess_output()
        self.stop_prediction_time()

    def inference(self):
        pass

    def preprocess_input(self):
        pass

    def preprocess_output(self):
        pass

red = (255, 0, 0)
green = (0, 255, 0)
blue = (0, 0, 255)

def round_point(p):
    return (int(p[0] + 0.5), int(p[1] + 0.5))

class FaceDetection:

    def __init__(self, detection):
        self.prob = detection[2]
        self.x0 = detection[3]
        self.y0 = detection[4]
        self.x1 = detection[5]
        self.y1 = detection[6]
        self.size = (np.abs(self.x1 - self.x0), np.abs(self.y1 - self.y0))
        self.area = self.size[0] * self.size[1]

    def __str__(self):
        return 'FaceDetection: prob={:.4f}, ({:.4f}, {:.4f}), ({:.4f}, {:.4f}), area={:.4f}'.format(self.prob, self.x0, self.y0, self.x1, self.y1, self.area)

    def compute_points(self, image):
        size = image.shape[:2]
        self.p0 = round_point((self.x0 * size[1], self.y0 * size[0]))
        self.p1 = round_point((self.x1 * size[1], self.y1 * size[0]))
        self.center = (self.p0[0] + self.size[0] * size[1] * 0.5, self.p0[1] + self.size[1] * size[0] * 0.5)

    def draw_rectangle(self, image):
        cv2.rectangle(image, self.p0, self.p1, green, thickness = 2)

    def crop_image(self, image):
        return image[self.p0[1] : self.p1[1], self.p0[0] : self.p1[0]].copy()

class FaceDetectionModel(AbstractModel):

    def inference(self):
        self.start_inference_time()
        self.objects2 = self.neural_network.infer({'data': self.input_image})
        self.stop_inference_time()

    def preprocess_input(self):
        self.input_shape = self.model.inputs['data'].shape
        self.input_image = cv2.resize(self.image, (self.input_shape[3], self.input_shape[2])).transpose((2,0,1))[None, :]

    def preprocess_output(self):
        detections = self.objects2['detection_out']
        self.fds = []
        for i in range(detections.shape[2]):
            detection = detections[0, 0, i]
            prob = detection[2]
            if prob < self.prob_threshold: break
            self.fds.append(FaceDetection(detection))
        self.fds = sorted(self.fds, key = lambda fd: fd.area, reverse = True)
        self.output_image = self.image.copy()
        for fd in self.fds: 
            fd.compute_points(self.output_image)
            if self.main_app.args.show_face_detection: fd.draw_rectangle(self.output_image)
            if self.main_app.args.verbose: logging.info('fd: {}'.format(fd))
        if len(self.fds) > 0: self.face_image = self.fds[0].crop_image(self.image)

class FacialLandmarks:

    def __init__(self, landmarks, fd):
        self.landmarks = landmarks
        self.n_landmarks = self.landmarks.shape[0] // 2
        self.fd = fd

    def __str__(self):
        s = ''
        for i in range(self.n_landmarks):
            s += '({:.4f},{:.4f}) '.format(self.landmarks[i * 2], self.landmarks[i * 2 + 1])
        return s

    def compute_point(self, lx, ly):
        x = self.fd.p0[0] + lx * self.size[0]
        y = self.fd.p0[1] + ly * self.size[1]
        return round_point((x, y))

    def draw_landmarks(self, image):
        for i in range(self.n_landmarks):
            center = self.compute_point(self.landmarks[i * 2], self.landmarks[i * 2 + 1])
            cv2.circle(image, center, radius = 8, color = green, thickness = 4) 

    def crop_eyes(self, image, s):
        def crop_eye(eye):
            x = eye[0] - s // 2
            y = eye[1] - s // 2
            return image[y : y + s, x : x + s]
        self.size = (self.fd.size[0] * image.shape[1], self.fd.size[1] * image.shape[0])
        self.eye0 = self.compute_point(self.landmarks[0], self.landmarks[1])
        self.eye1 = self.compute_point(self.landmarks[2], self.landmarks[3])
        return (crop_eye(self.eye0), crop_eye(self.eye1))

class FacialLandmarkModel(AbstractModel):

    def inference(self):
        self.start_inference_time()
        self.objects2 = self.neural_network.infer({'0': self.input_image})
        self.stop_inference_time()

    def preprocess_input(self):
        self.input_shape = self.model.inputs['0'].shape
        self.input_image = cv2.resize(self.face_image, (self.input_shape[3], self.input_shape[2])).transpose((2,0,1))[None, :]

    def preprocess_output(self):
        self.landmarks = FacialLandmarks(self.objects2['95'][0, :, 0, 0], self.fds[0])
        s = 60
        self.eye0, self.eye1 = self.landmarks.crop_eyes(self.image, s)
        if self.main_app.args.show_facial_landmarks: 
            self.landmarks.draw_landmarks(self.output_image)
            self.output_image[:s, :s] = self.eye0
            self.output_image[:s, s:s+s] = self.eye1
        if self.main_app.args.verbose: logging.info('landmarks: {}'.format(self.landmarks))

class HeadPose:

    def __init__(self, main_app, objects2):
        self.main_app = main_app
        self.yaw = objects2['angle_y_fc'][0, 0]
        self.pitch = objects2['angle_p_fc'][0, 0]
        self.roll = objects2['angle_r_fc'][0, 0]        

    def __str__(self):
        return 'yaw={:.4f}, pitch={:.4f}, roll={:.4f}'.format(self.yaw, self.pitch, self.roll)

    def to_numpy(self):
        return np.array([self.yaw, self.pitch, self.roll])

    def to_radians(self):
        return self.to_numpy() / 180.0 * np.pi

    # https://math.stackexchange.com/questions/1637464/find-unit-vector-given-roll-pitch-and-yaw
    def compute_rotated_base(self):
        yaw, pitch, roll = tuple(self.to_radians())
        roll *= -1
        #if self.main_app.args.input_type == 'cam': yaw *= -1
        sr = np.sin(roll)
        cr = np.cos(roll)
        roll_matrix = np.array([[cr, 0, -sr], [0, 1, 0], [sr, 0, cr]])
        sp = np.sin(pitch)
        cp = np.cos(pitch)
        pitch_matrix = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
        sy = np.sin(yaw)
        cy = np.cos(yaw)
        yaw_matrix = np.array([[cy, sy, 0], [-sy, cy, 0], [0, 0, 1]])
        i = np.array([1, 0, 0])
        j = np.array([0, -1, 0])
        k = np.array([0, 0, 1])
        i2 = i.dot(roll_matrix).dot(pitch_matrix).dot(yaw_matrix)
        j2 = j.dot(roll_matrix).dot(pitch_matrix).dot(yaw_matrix)
        k2 = k.dot(roll_matrix).dot(pitch_matrix).dot(yaw_matrix)
        return i2, j2, k2

    def draw_line(self, image, fd, v, color, arrow_length, thickness):
        v2 = v * arrow_length
        p0 = round_point((fd.center[0], fd.center[1]))
        #p1 = round_point((fd.center[0] + v2[0], fd.center[1] + v2[1]))
        p1 = round_point((fd.center[0] + v2[0], fd.center[1] + v2[2]))
        cv2.line(image, p0, p1, color, thickness)

    def draw_rotated_base(self, image, fd):
        i, j, k = self.compute_rotated_base()
        self.draw_line(image, fd, i, red, 100, 2)
        self.draw_line(image, fd, -i, red, 100, 2)
        self.draw_line(image, fd, k, blue, 100, 2)
        self.draw_line(image, fd, -k, blue, 100, 2)
        self.draw_line(image, fd, j, green, 200, 4)

class HeadPoseModel(AbstractModel):

    def inference(self):
        self.start_inference_time()
        self.objects2 = self.neural_network.infer({'data': self.input_image})
        self.stop_inference_time()

    def preprocess_input(self):
        self.input_shape = self.model.inputs['data'].shape
        self.input_image = cv2.resize(self.face_image, (self.input_shape[3], self.input_shape[2])).transpose((2,0,1))[None, :]

    def preprocess_output(self):
        self.pose = HeadPose(self.main_app, self.objects2)
        if self.main_app.args.show_head_pose: self.pose.draw_rotated_base(self.output_image, self.fds[0])
        if self.main_app.args.verbose: logging.info('pose: {}'.format(self.pose))

class GazeEstimationModel(AbstractModel):

    def inference(self):
        self.start_inference_time()
        objects = {
            'left_eye_image': self.left_eye_image,
            'right_eye_image': self.right_eye_image,
            'head_pose_angles': self.head_pose_angles[None, :]
        }
        self.objects2 = self.neural_network.infer(objects)
        self.stop_inference_time()

    def preprocess_input(self):
        self.left_eye_image = self.left_eye_image.transpose((2,0,1))[None, :]
        self.right_eye_image = self.right_eye_image.transpose((2,0,1))[None, :]

    def preprocess_output(self):
        self.gaze_vector = self.objects2['gaze_vector'][0]
        if self.main_app.args.show_gaze_estimation: 
            self.draw_line(self.output_image, self.landmarks.eye0, self.gaze_vector, green, 100, 4)
            self.draw_line(self.output_image, self.landmarks.eye1, self.gaze_vector, green, 100, 4)
        if self.main_app.args.verbose: logging.info('gaze_vector={}'.format(self.gaze_vector))

    def draw_line(self, image, center, v, color, arrow_length, thickness):
        v2 = v * arrow_length
        p0 = round_point((center[0], center[1]))
        p1 = round_point((center[0] + v2[0], center[1] - v2[1]))
        cv2.line(image, p0, p1, color, thickness)

