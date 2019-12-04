#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 15:29:54 2019

@author: Federica Lareno Faccini (modified from Adafruit)

-------------------------------------------------------------------------------

Sampling rate: 10 Hz (max allowed by the camera's hardware)
The script initializes the sensor and captures frames of the thermal sensor.
The sensor has 64 pixels (8x8) but the resolution is increased by means of bicubic interpolation of the pixels.
The pictures are saved in the savedire in a sequential manner (incremental number).
The recording is triggered by the rising edge of a TTL and continues recording for x = 'timeout' seconds.

"""
import math
import time
import busio
import board
import numpy as np
import pygame
from scipy.interpolate import griddata
from colour import Color

import adafruit_amg88xx
import RPi.GPIO as GPIO


###############################################################################
###########    CAMERA SET-UP (MODIFY AT YOUR OWN RISK!!)    ###################
###############################################################################

i2c_bus = busio.I2C(board.SCL, board.SDA)

#low range of the sensor (this will be blue on the screen)
MINTEMP = 26.
#high range of the sensor (this will be red on the screen)
MAXTEMP = 32.
#how many color values we can have
COLORDEPTH = 1024

#initialize the sensor
sensor = adafruit_amg88xx.AMG88XX(i2c_bus, addr=0x69)

#Temperature in Celsius of every pixel (8x8)
#First row is the closest to the wiring
#a = sensor.pixels

# creates an array with the gridpoints
points = [(math.floor(ix / 8), (ix % 8)) for ix in range(0, 64)]
#the step length is a COMPLEX NUMBER (32j) so the integer part is interpreted as specifying the number of points to create between start and stop. Stop being inclusive.
grid_x, grid_y = np.mgrid[0:7:32j, 0:7:32j] 

#sensor is an 8x8 grid so lets do a square
height = 720
width = 720

#the list of colors we can choose from
blue = Color("indigo")
colors = list(blue.range_to(Color("red"), COLORDEPTH)) # To create a list of shades between two colors!

#create the array of colors
colors = [(int(c.red * 255), int(c.green * 255), int(c.blue * 255)) for c in colors] # There are 3 elements because it's RGB. 255 is used because it's the span to get white (255,255,255)

displayPixelWidth = width / 30
displayPixelHeight = height / 30

lcd = pygame.display.set_mode((width, height))
pygame.mouse.set_visible(False)
lcd.fill((0, 0, 0))
pygame.display.update()

    
#some utility functions
def constrain(val, min_val, max_val):
    return min(max_val, max(min_val, val))

def map_value(x, in_min, in_max, out_min, out_max):
    return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min

###############################################################################
###############################################################################
    

savedir = '/home/pi/Pictures/Thermal_Cam/'

mouse = 1089  
session = 1      


# TTL detection
ttl_in = 26
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)
GPIO.setup(ttl_in, GPIO.IN, pull_up_down = GPIO.PUD_DOWN)

trial = 1
while 1:
    
    try:
        trial += 1
        GPIO.wait_for_edge(ttl_in, GPIO.RISING)
            
       #let the sensor initialize
    #    time.sleep(.1)
        
        # Initialize name of the file
        _image_num = 000
        
        timeout = 4   # [seconds]
        timeout_start = time.time()
    
        while time.time() < timeout_start + timeout:
            
            #Increment name of the file
            _image_num += 1
            
            ###################################################################
            ###################   WHERE THE MAGIC HAPPENS   ###################
            ###################################################################
            
            #read the pixels
            pixels = []
            for row in sensor.pixels:
                pixels = pixels + row # Creates a list with all the values of the 64 pixels. this loop is repeated and every 8 loops we have an array with all 64 pixels in a row
            pixels = [map_value(p, MINTEMP, MAXTEMP, 0, COLORDEPTH - 1) for p in pixels] # measures, for each pixel, the relative temperature of the pixel and gives the corresponding color value in the range
        
            #perform interpolation
            bicubic = griddata(points, pixels, (grid_x, grid_y), method='cubic')
        
            #draw everything
            for ix, row in enumerate(bicubic):
                for jx, pixel in enumerate(row):
                    pygame.draw.rect(lcd, colors[constrain(int(pixel), 0, COLORDEPTH- 1)],
                                     (displayPixelHeight * ix, displayPixelWidth * jx,
                                      displayPixelHeight, displayPixelWidth))            
            
            pygame.image.save(lcd, savedir+'{}_{}_{}_{:06d}.jpg'.format(mouse, session, trial, _image_num))  
                     
            pygame.display.update()
            
            ###################################################################
            ###################################################################

    except KeyboardInterrupt:
        print('Camera ended')
        pygame.quit()
               
if KeyboardInterrupt:
    print('Camera ended')
    pygame.quit()