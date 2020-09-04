'''
This class can be used to feed input from an image, webcam, or video to your model.
Sample usage:
    feed=InputFeeder(input_type='video', input_file='video.mp4')
    feed.load_data()
    for batch in feed.next_batch():
        do_something(batch)
    feed.close()
'''
import cv2
from numpy import ndarray
import sys
import logging

class InputFeeder:

    def __init__(self, main_app, input_type, input_file = None):
        '''
        input_type: str, The type of input. Can be 'video' for video file, 'image' for image file,
                    or 'cam' to use webcam feed.
        input_file: str, The file that contains the input image or video file. Leave empty for cam input_type.
        '''
        self.main_app = main_app
        self.input_type = input_type
        if input_type == 'video' or input_type == 'image':
            self.input_file = input_file
        elif input_type == 'cam':
            self.input_file = None
        else:
            logging.error('\n"{}" is not a supported input type. Supported input types are "video", "cam", or "image".\n'.format(self.input_type))
            sys.exit(0)
    
    def load_data(self):
        if self.input_type == 'video':
            self.cap = cv2.VideoCapture(self.input_file)
        elif self.input_type == 'cam':
            self.cap = cv2.VideoCapture(0)
        else:
            self.cap = cv2.imread(self.input_file)

    def next_batch(self):
        '''
        Returns the next image from either a video file or webcam.
        If input_type is 'image', then it returns the same image.
        '''
        if self.input_type == 'image':
            while True:
                yield False, self.cap
        while True:
            ret, frame = self.cap.read()
            yield ret, frame

    def close(self):
        '''
        Closes the VideoCapture.
        '''
        if not self.input_type == 'image':
            self.cap.release()

    def get_size(self):
        width = int(self.cap.get(3))
        height = int(self.cap.get(4))
        return (width, height)

    def get_frame_rate(self):
        return self.cap.get(cv2.CAP_PROP_FPS)

    def get_video_writer(self, output_file, fourcc_string = 'mp4v'):
        frame_rate = self.get_frame_rate()
        size = self.get_size()
        fourcc = cv2.VideoWriter_fourcc(*fourcc_string)
        logging.info('CREATING VIDEO WRITER: output_file={}, frame_rate={:.2f}, size={}, fourcc_string={}'.format(output_file, frame_rate, size, fourcc_string))
        return cv2.VideoWriter(output_file, fourcc, frame_rate, size)

