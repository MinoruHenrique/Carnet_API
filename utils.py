import cv2
import numpy as np
import pickle
def load_spaces(folder_dir):
    spaces = []
    with open(folder_dir, "r") as f:
        for line in f:
            coords = line.split(" ")
            A = [int(coords[0]), int(coords[1])]
            B = [int(coords[2]),int(coords[3])]
            C = [int(coords[4]), int(coords[5])]
            D = [int(coords[6]), int(coords[7][:-1])]
            spaces.append([A,B,C,D])
    return spaces

def crop_from_space(img, space):
    pt_A,pt_B,pt_C,pt_D = space
    
    WIDTH = 128
    HEIGHT = 216
    
    input_pts = np.float32([pt_A, pt_B, pt_C, pt_D])
    output_pts = np.float32([[0, 0],
                        [WIDTH-1, 0],
                        [WIDTH-1, HEIGHT-1],
                        [0, HEIGHT-1]])
    
    M = cv2.getPerspectiveTransform(input_pts,output_pts)
    cropped_img = cv2.warpPerspective(img,M,(WIDTH,HEIGHT))
    return cropped_img
def capture_img(cap, cameraMatrix,dist):
    ret, frame = cap.read()
    h,w = frame.shape[:2]
    newCameraMatrix, roi = cv2.getOptimalNewCameraMatrix(cameraMatrix, dist, (w,h), 1, (w,h))
    dst = cv2.undistort(frame, cameraMatrix, dist, None, newCameraMatrix)

    # crop the image
    x, y, w, h = roi
    dst = dst[y:y+h, x:x+w]
    dst = cv2.resize(dst, (1280,720))
    return dst
def capture_spaces(frame, spaces):   
    cropped_imgs = []
    test_list = []
    for space in spaces:
        cropped_img = crop_from_space(frame, space)
        test_list.append(cropped_img)
        cropped_img = cv2.resize(cropped_img, (32,54))/255
        space = np.array(space).reshape(-1,1,2)
        cropped_imgs.append(cropped_img)
    cropped_imgs = np.array(cropped_imgs)
    return cropped_imgs, test_list

def load_pickle_file(file):
    with open(file,'rb') as f:
        unpickled_file = pickle.load(f)
    return unpickled_file