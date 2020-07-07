
# -*- coding: utf-8 -*-
"""
Created on Tue May 02 17:16:33 2017

@author: Federica Lareno Faccini
"""

import picamera

camera=picamera.PiCamera()
camera.annotate_background = True
camera.resolution=(640, 480)

input("PRESS ENTER TO START PREVIEW")

camera.start_preview()
protocol_in_execution=1
first_time = 1
keep_parameters = 0

input("PRESS ENTER TO STOP PREVIEW")

camera.stop_preview()


