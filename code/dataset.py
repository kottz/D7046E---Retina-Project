import cv2
import numpy as np
import json
import random

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# Set default figure size
plt.rcParams['figure.figsize'] = [6,6]

# Function that is used to plot spike times
def rasterplot(ax, x, y, x_label, y_label):
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.scatter(x, y, marker='|')
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))

# background

# 

#cap = cv2.VideoCapture('man_walking.mp4')
init = False
prev_frame = None
x = 150
i = 0
'''
while(True):
    # Capture frame-by-frame

    ret, frame = cap.read()
    
    frame = cv2.flip(frame,1)

    # Our operations on the frame come here
    

    # Display the resulting frame
    


    #1) Remove green pixels from image
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_green = np.array([40,40,40])
    upper_green = np.array([70,255,255])

    u_green = np.array([104, 153, 70]) 
    l_green = np.array([30, 30, 0]) 

    mask = cv2.inRange(hsv, lower_green, upper_green)
    
    res = cv2.bitwise_and(frame, frame, mask = mask) 
  
    f = frame - res 
    #f = res
    h, s, gray = cv2.split(f)
    #f = np.where(f == 0, image, f) 


    #cv2.imshow('frame',frame)
    #cv2.imshow('mask',mask)
    #cv2.imshow('res',res)

    cv2.imshow('gray',gray)

    #cv2.imshow('f',f)
    #1) Translate image to move object (With different parameters)
    rows,cols = gray.shape

    dx = -1
    y = 0

    x += dx
    M = np.float32([[1,0,x],[0,1,y]])
    dst = cv2.warpAffine(gray,M,(cols,rows))
    cv2.imshow('dst',dst)

    #2) Cut out a horizontal slice
    crop_img = gray[180:181, 0:360]
    cv2.imshow("cropped", crop_img)

    #3) Determine what constitutes a spike. Difference in intensity from previous frame.
    #prev_frame compare to current frame
    #diff = prev_frame - frame
    
    #4) Export data in some way (video files or spike arrays?)
    if not init:
        prev_dst = dst
        init = True


    
    #diff = prev_f - f
    diff = dst - prev_dst
    prev_dst = dst
    cv2.imshow('diff',diff)
    


    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
'''
def get_video_format(video_src):
    cap = cv2.VideoCapture(video_src)
    ret, frame = cap.read()
    height, width, _ = frame.shape
    cap.release()
    return (height, width)

SHOW_GUI = True

def create_video(video_src, starting_offset, speed, speed_mode, flip, cropped_pixel_height, dt):
    # Op
    height, width = get_video_format(video_src)


    cap = cv2.VideoCapture(video_src)
    #ret, prev_frame = cap.read()
    x = starting_offset
    init = False
    spikes = []
    t = 0
    for i in range(width):
        spikes.append([])

    while(True):
        ret, frame = cap.read()
        
        #if frame.empty():
         #   break
        if not ret:
            break
        if flip:
            frame = cv2.flip(frame,1)

    
        #Convert to HSV color space so that we can extract green parts more easily
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_green = np.array([40,40,40])
        upper_green = np.array([70,255,255])
        mask = cv2.inRange(hsv, lower_green, upper_green) 

        #
        res = cv2.bitwise_and(frame, frame, mask = mask) 
        #Extract only the non-green parts of the video (f now contains the moving)
        f = frame - res 
        # extract intensity levels from hsv frame
        h, s, intensity = cv2.split(f)

        # Translate video (by speed pixels)

        # How to handle different speed setting modes? Some type of function parameter

        #Constant speed
        if speed_mode == "CONSTANT":
            dx = speed
        elif speed_mode == "NORMAL":
            dx = generate_speed(speed)
        else:
            #Default to constant mode
            dx = speed
        # Normal distribution per frame
        

        rows,cols = intensity.shape
        x += dx
        M = np.float32([[1,0,x],[0,1,0]])
        translated_intensity = cv2.warpAffine(intensity,M,(cols,rows))

        # Set current frame as prev frame first run
        if not init:
                prev_translated_intensity = translated_intensity
                init = True

        diff = translated_intensity - prev_translated_intensity
        prev_translated_intensity = translated_intensity
        #cv2.imshow('diff',diff)

        # Extract 1D slice of pixels
        cropped_intensity = translated_intensity[cropped_pixel_height:cropped_pixel_height+1, 0:width]
        #cv2.imshow("cropped", cropped_intensity)

        cropped_intensity[cropped_intensity > 0] = 1

        all_zeros = not cropped_intensity.any()
        
        if all_zeros:
            continue
        #print("---")
        #print(cropped_intensity)
        #print("---")


        for i in range(len(cropped_intensity[0])):
            if cropped_intensity[0][i] == 1:
                spikes[i].append(t)
        
        t += dt
 
        if SHOW_GUI:
            #cv2.imshow('frame',frame)
            cv2.imshow('diff',diff)
            if cv2.waitKey(33) & 0xFF == ord('q'):
                break
    cap.release()
    return spikes

def list_only_empty(l):
    for x in l:
        if x != []:
            return False
    return True

def clean_data(d):
    i = 0
    while i < len(d):
        s = d[i]['spikes']
        if list_only_empty(s):
            d.pop(i)
        else:
            i += 1

def generate_speed(mu):
    s = np.random.normal(mu, 2)
    if s < 0:
        s = 0
    return round(s)

spikes = create_video('man_walking.mp4', 250, -1, "CONSTANT", True, 180, 0.033333)
print(spikes)
#spikes = create_video('man_walking.mp4', -250, 10, "NORMAL", False, 180, 0.03333)
#print(spikes)


#create_video('man_walking.mp4', 250, -3, True, 180)

# Left
#for pixel_height in range(180,185):
#    for speed in range(2):
#        create_video('man_walking.mp4', -250, speed, False, pixel_height)
# When everything done, release the capture

data = {}
data['right'] = []
data['left'] = []
'''
data['right'].append({
    'source_file': "man_walking.mp4",
    'horizontal_offset': -250,
    'speed': 3,
    'mirrored': False,
    'vertical_slice': 180,
    'delta_t': 33.333,
    'spikes': spikes
})
'''



input_videos = ['man_walking_60p.mp4']

dt = 0.03333

# Right

SHOW_GUI = False

for video in input_videos:
    height, width = get_video_format(video)
    horizontal_offset = -width
    i = 0
    for pixel_height in range(0, height):
        print("Calculating height: {}/{}".format(str(i), str(height)))
        i +=1
        #Constant speed 1-x
        for speed in range(1,5):
            spikes = create_video(video, horizontal_offset, speed, "CONSTANT", False, pixel_height, dt)
            data['right'].append({
                'source_file': video,
                'horizontal_offset': horizontal_offset,
                'speed': speed,
                'mirrored': False,
                'vertical_slice': pixel_height,
                'delta_t': dt,
                'spikes': spikes
            })
        #Normal distribution speed 1-x
        for _ in range(10):
            mu_speed = int(np.random.normal(4, 3))
            spikes = create_video(video, horizontal_offset, mu_speed, "NORMAL", False, pixel_height, dt)
            data['right'].append({
                'source_file': video,
                'horizontal_offset': horizontal_offset,
                'mu_speed': mu_speed,
                'mirrored': False,
                'vertical_slice': pixel_height,
                'delta_t': dt,
                'spikes': spikes
            })


# Left
for video in input_videos:
    height, width = get_video_format(video)
    horizontal_offset = width
    i = 0
    for pixel_height in range(0, height):
        print("Calculating height: {}/{}".format(str(i), str(height)))
        i +=1
        for speed in range(1,5):
            spikes = create_video(video, horizontal_offset, -speed, "CONSTANT", True, pixel_height, dt)
            data['left'].append({
                'source_file': video,
                'horizontal_offset': horizontal_offset,
                'speed': -speed,
                'mirrored': True,
                'vertical_slice': pixel_height,
                'delta_t': dt,
                'spikes': spikes
            })
        #Normal distribution speed 1-x
        for _ in range(10):
            mu_speed = int(np.random.normal(4, 3))
            spikes = create_video(video, horizontal_offset, -mu_speed, "NORMAL", True, pixel_height, dt)
            data['left'].append({
                'source_file': video,
                'horizontal_offset': horizontal_offset,
                'mu_speed': -mu_speed,
                'mirrored': True,
                'vertical_slice': pixel_height,
                'delta_t': dt,
                'spikes': spikes
            })




#print("---")
#print(data)



#for x in data['left']:
#    s = x['spikes']
#    if list_only_empty(s):





clean_data(data['right'])
clean_data(data['left'])

print("number of elements after cleaning. Left: {}, Right: {}".format(str(len(data['left'])), str(len(data['right']))))
    

# TODO: Remove empty entries. (If all spike arrays are empty)
# Kolla intensitetsintervall.

with open("data_file.json", "w") as write_file:
    json.dump(data, write_file)


# Ideas to generate more data from 60p video
# - Stretch video, increase height and keep width at 60
# - Pick speed from a random distribution. Some frames are slow, some are fast.
# - Vary the initial horizontal offset, start in the middle of the screen in some runs.


cv2.destroyAllWindows()