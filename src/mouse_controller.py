'''
This is a sample class that you can use to control the mouse pointer.
It uses the pyautogui library. You can set the precision for mouse movement
(how much the mouse moves) and the speed (how fast it moves) by changing 
precision_dict and speed_dict.
Calling the move function with the x and y output of the gaze estimation model
will move the pointer.
This class is provided to help get you started; you can choose whether you want to use it or create your own from scratch.
'''
import pyautogui
# https://pyautogui.readthedocs.io/en/latest/quickstart.html
import logging

class MouseController:
    def __init__(self, main_app, precision = 40, duration_seconds = 0):
        self.main_app = main_app
        self.precision = precision
        self.duration_seconds = duration_seconds
        pyautogui.FAILSAFE = False
        self.screen_size = pyautogui.size()
        logging.info('screen_size: {}'.format(self.screen_size))
        self.center_pointer()
        
    def center_pointer(self):
        pyautogui.moveTo(self.screen_size[0] // 2, self.screen_size[1] // 2)

    def move(self, x, y):
        horizontal_correction = 1
        #horizontal_correction = -1 if self.main_app.args.input_type == 'cam' else 1
        pyautogui.moveRel(horizontal_correction * x * self.precision, -y * self.precision, duration = self.duration_seconds)

    def get_position(self):
        return pyautogui.position()

