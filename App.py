import toga
from toga.style import Pack
from toga.style.pack import COLUMN, ROW
import cv2
import os
import numpy as np
import random
import imageio
from scipy.fft import fft, fftfreq, ifft
from sklearn.metrics import auc
import io
from PIL import Image, ImageDraw
from toga.style.pack import CENTER, COLUMN, Pack
from toga.colors import WHITE, rgb
from toga.constants import Baseline
from toga.fonts import SANS_SERIF
from toga.style import Pack
import math


class StartApp(toga.App):
    count = 0
    def startup(self):
        print(self.paths.app,"####")
        self.main_window = toga.MainWindow(size=(150, 250))
        
        # Create empty canvas
        self.canvas = toga.Canvas(
            style=Pack(flex=1),
            on_resize=self.on_resize,
            on_press=self.on_press,
        )
        box = toga.Box(children=[self.canvas])
        self.main_window.content = box
        image_from_path = toga.Image("S__11444239.jpg")
        # First display the image at its intrinsic size.
        b1 = box.add(
            toga.ImageView(
                image_from_path,
                #style=Pack(flex=1)
            )
        )

        ####
        

        #path = "./resources/"
        print("self.paths:",self.paths.app)
        os.chdir(self.paths.app)
        path = str(self.paths.app) + '/resources/'
        haarcascade = cv2.CascadeClassifier(path + "haarcascade_frontalface_alt2.xml")
        LBFmodel = path +"lbfmodel.yaml"
        print("@@@@@@, haarcas loded")
        #haarcascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')
        landmark_detector  = cv2.face.createFacemarkLBF()
        landmark_detector.loadModel(LBFmodel)
        print("@@@@@@, lbf loded")
        def convex(src_img, raw, effect):
            col, row, channel = raw[:]      
            cx, cy, r = effect[:]           
            output = np.zeros([row, col, channel], dtype = np.uint8)      
            for y in range(row):
                for x in range(col):
                    d = ((x - cx) * (x - cx) + (y - cy) * (y - cy)) ** 0.5  
                    if d <= r:
                        nx = int((x - cx) * d / r + cx)        
                        ny = int((y - cy) * d / r + cy)        
                        output[y, x, :] = src_img[ny, nx, :]   
                    else:
                        output[y, x, :] = src_img[y, x, :]     
            return output
        def paste_png(img, fu, harm, dirc, eyelx,eyely,eyerx,eyery): # dirc: left or right
            c1x,c1y,c2x,c2y = eyelx,eyely,eyerx,eyery
            ma_x1_width = 100
            ma_y1_width = 100
        #print(ma_x1_width, ma_y1_height)
            ran_x1_width, ran_y1_width =  random.randint(int(ma_x1_width/2),int(ma_x1_width)), \
            random.randint(int(ma_y1_width/2),int(ma_y1_width))
            ba = img
            if dirc == 'l':
        #print("###")
                c1x_start, c1y_start = int(np.mean(c1x)), int(np.mean(c1y))
            else:
                c1x_start, c1y_start = int(np.mean(c2x)), int(np.mean(c2y))
            #print(c1x_start,c1y_start)
            shift = 30
            c1x_start -= shift
            #print(int(c1y_start),int(c1y_start+ran_y1_width))
            #print("############",ba)
            background = ba[int(c1y_start):int(c1y_start+ran_y1_width),
                    int(c1x_start):int(c1x_start+ran_x1_width)]

            overlay = cv2.imread(fu, cv2.IMREAD_UNCHANGED)  # IMREAD_UNCHANGED => open image with the alpha channel
            dim = (background.shape[1],background.shape[0])
            overlay = cv2.resize(overlay,dim , interpolation = cv2.INTER_AREA)
            height, width = overlay.shape[:2]
            print(background.shape,overlay.shape)
            for y in range(height):
                for x in range(width):
                    overlay_color = overlay[y, x, :3]  # first three elements are color (RGB)
                    overlay_alpha = overlay[y, x, 3] / 255  # 4th element is the alpha channel, convert from 0-255 to 0.0-1.0

            # get the color from the background image
                    background_color = background[y, x]

            # combine the background color and the overlay color weighted by alpha
                    composite_color = background_color * (1 - overlay_alpha) + overlay_color * overlay_alpha

            # update the background image in place
                    background[y, x] = composite_color
            ba[int(c1y_start):int(c1y_start+ran_y1_width),int(c1x_start):int(c1x_start+ran_x1_width)]
    #for i in range(15):
            ranx = random.randint(int(c1x_start),int(c1x_start+ran_x1_width))
            rany = random.randint(int(c1y_start),int(c1y_start+ran_y1_width))
    #test = convex(ba, (ba.shape[1], ba.shape[0], 3), 
    #(int((c1y_start+ran_y1_height//2)), int((c1x_start+ran_x1_width//2)),  (i+1)*5))
            ba = convex(ba, (ba.shape[1], ba.shape[0], 3), 
            (ranx, rany,  harm*5))
            #cv2.imshow('image',ba)
            #cv2.waitKey(0)                 
            #cv2.destroyAllWindows()
            return ba
        def find_eye(img):
            img = cv2.imread(img)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = haarcascade.detectMultiScale(gray,1.3,5)
        #print(faces)
            _, landmarks = landmark_detector.fit(gray, faces)
        #eye = []
            reye = landmarks[0][0][36:41]
            leye = landmarks[0][0][42:48]
            reye_center_x, reye_center_y,leye_center_x, leye_center_y = 0, 0, 0, 0
            for i in range(len(reye)):
                reye_center_x += reye[i][0]
                reye_center_y += reye[i][1]
            reye_center_x =  int(reye_center_x // len(reye))
            reye_center_y =  int(reye_center_y // len(reye))
    
            for i in range(len(leye)):
                leye_center_x += leye[i][0]
                leye_center_y += leye[i][1]
            leye_center_x =  int(leye_center_x // len(leye))
            leye_center_y =  int(leye_center_y // len(leye))
            print(reye_center_x,reye_center_y,leye_center_x,leye_center_y)

            return reye_center_x,reye_center_y,leye_center_x,leye_center_y
        ####
        print("###prepare to find eye")
        eyelx,eyely,eyerx,eyery = find_eye(path+"S__11444239.jpg")

        
        def left_punch(none):
            new_img = 'test'+str(StartApp.count)+'.jpg'
            print(StartApp.count,"_@@@@@@@@@@@@@@")
            harm,dirc = 5, 'l'
            if StartApp.count<=0:
                img = cv2.imread(path+ "S__11444239.jpg" )
            else:
                #print("#################",path+ new_img )
                next_img = 'test'+str(StartApp.count-1)+'.jpg'
                img = cv2.imread(path+ next_img )
            fu = path+"fu1.png"
            ret = paste_png(img, fu, harm, dirc, eyelx,eyely,eyerx,eyery)
            cv2.imwrite(path+new_img, ret)
            new_box = toga.Box()
            new_box.add(
            toga.ImageView(
                path + new_img,
                #style=Pack(flex=1)
            )
        )
            #print(path)
            #gen = toga.Image(path +new_img)
            button = toga.Button("Calculate", 
            on_press= left_punch)
            new_box.add(button)
            self.main_window.content = new_box
            StartApp.count += 1


            #print(ret)
        # Add the content on the main window


        button = toga.Button("Calculate", 
            on_press= left_punch)
        box.add(button)
        ###
        self.main_window.show()

    def on_resize(self, widget, width, height, **kwargs):
        # On resize, center the text horizontally on the canvas. on_resize will be
        # called when the canvas is initially created, when the drawing objects won't
        # exist yet. Only attempt to reposition the text if there's context objects on
        # the canvas.
        if widget.context:
            left_pad = (width - self.text_width) // 2
            self.text.x = left_pad
            self.text_border.x = left_pad - 5
            widget.redraw()

    def on_press(self, widget, x, y, **kwargs):
        self.main_window.info_dialog("Hey!", f"You poked the yak at ({x}, {y})")


def main():
    return StartApp("Tutorial 4", "org.beeware.helloworld")


if __name__ == "__main__":
    main().main_loop()
