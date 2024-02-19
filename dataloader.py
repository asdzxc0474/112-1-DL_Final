import os
import random
import cv2
from PyQt5.QtGui import *

class random_load_testdata():
    def __init__(self, test_dir):
        self.test_dir = test_dir
        self.subdirectories = [d for d in os.listdir(self.test_dir) if os.path.isdir(os.path.join(self.test_dir, d))]
    def ramdom_out(self):    
        if not self.subdirectories:
            print("No subdirectories found in the test directory.")
        else:
            random_subdirectory = random.choice(self.subdirectories)
            random_subdirectory_path = os.path.join( self.test_dir, random_subdirectory)
            files_in_subdirectory = [f for f in os.listdir(random_subdirectory_path) if os.path.isfile(os.path.join(random_subdirectory_path, f))]
            if not files_in_subdirectory:
                print(f"No files found in the subdirectory {random_subdirectory}.")
            else:
                random_file = random.choice(files_in_subdirectory)
                random_file_path = os.path.join(random_subdirectory_path, random_file)
        return random_file_path  
def putext(img, preds):
    class_names = ['african_elephant', 'alpaca', 'american_bison', 'anteater', 'arctic_fox', 'armadillo',\
                        'baboon', 'badger', 'blue_whale', 'brown_bear', 'camel', 'dolphin', 'giraffe', 'groundhog',\
                            'highland_cattle', 'horse', 'jackal', 'kangaroo', 'koala', 'manatee', 'mongoose',\
                                'mountain_goat', 'opossum', 'orangutan', 'otter', 'polar_bear', 'porcupine',\
                                    'red_panda', 'rhinoceros', 'sea_lion', 'seal', 'snow_leopard', 'squirrel',\
                                        'sugar_glider', 'tapir', 'vampire_bat', 'vicuna', 'walrus', 'warthog',\
                                            'water_buffalo', 'weasel', 'wildebeest', 'wombat', 'yak', 'zebra']
    img = cv2.resize(img, (960,960))
    cv2.rectangle(img, (0,0), (400, 35), (255, 255, 255), -1, cv2.LINE_AA)
    cv2.putText(img,'preds:' + str(class_names[preds]), (0, 20), cv2.FONT_HERSHEY_TRIPLEX,1, (0, 0, 0), 2)
    return img
def cv2qimg(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  
    height, width, channel = img.shape          
    bytesPerline = channel * width             
    qimg = QImage(img, width, height, bytesPerline, QImage.Format_RGB888)
    return qimg