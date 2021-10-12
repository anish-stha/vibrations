import csv

import cv2
import numpy as np


def ssim(i1, i2):
    c1 = 6.5025
    c2 = 58.5225
    # INITS
    I1 = np.float32(i1) # cannot calculate on one byte large values
    I2 = np.float32(i2)
    I2_2 = I2 * I2 # I2^2
    I1_2 = I1 * I1 # I1^2
    I1_I2 = I1 * I2 # I1 * I2
    # END INITS
    # PRELIMINARY COMPUTING
    mu1 = cv2.GaussianBlur(I1, (11, 11), 1.5)
    mu2 = cv2.GaussianBlur(I2, (11, 11), 1.5)
    mu1_2 = mu1 * mu1
    mu2_2 = mu2 * mu2
    mu1_mu2 = mu1 * mu2
    sigma1_2 = cv2.GaussianBlur(I1_2, (11, 11), 1.5)
    sigma1_2 -= mu1_2
    sigma2_2 = cv2.GaussianBlur(I2_2, (11, 11), 1.5)
    sigma2_2 -= mu2_2
    sigma12 = cv2.GaussianBlur(I1_I2, (11, 11), 1.5)
    sigma12 -= mu1_mu2
    t1 = 2 * mu1_mu2 + c1
    t2 = 2 * sigma12 + c2
    t3 = t1 * t2                    # t3 = ((2*mu1_mu2 + C1).*(2*sigma12 + C2))
    t1 = mu1_2 + mu2_2 + c1
    t2 = sigma1_2 + sigma2_2 + c2
    t1 = t1 * t2                    # t1 =((mu1_2 + mu2_2 + C1).*(sigma1_2 + sigma2_2 + C2))
    ssim_map = cv2.divide(t3, t1)    # ssim_map =  t3./t1;
    mssim = cv2.mean(ssim_map)       # mssim = average of ssim map
    return mssim


cap = cv2.VideoCapture("no_vibrations.mp4")

count = 0
frm1 = -1

cv2.namedWindow("frame-1")
cv2.moveWindow("frame-1", 800, 0)

total_ssim = 0



# writing to csv file
with open("no_vibration_difference", 'w') as csvfile:
    # creating a csv writer object
    csvwriter = csv.writer(csvfile)

    while cap.isOpened():
        ret, frm = cap.read()

        if ret:
            if count == 0:
                frm1 = frm
            else:
                pixel = frm[700, 400]
                pixel_next = frm1[900, 900]
                print([pixel_next, pixel, [int(pixel_next[0]) - int(pixel[0]), int(pixel_next[1]) - int(pixel[1]), int(pixel_next[2]) - int(pixel_next[2])]])
                # writing the fields
                csvwriter.writerow([int(pixel_next[0]) - int(pixel[0]), int(pixel_next[1]) - int(pixel[1]), int(pixel_next[2]) - int(pixel_next[2])])
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            count += 1
        else:
            break

print("===============================================================")

cap.release()
cv2.destroyAllWindows()