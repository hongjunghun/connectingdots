import os
import cv2
import numpy as np
import math

folder_path = "D:\\pythonworks\\automatic\\dot\\image\\targets"
folder_working_path = "D:\\pythonworks\\automatic\\dot\\image\\results"

os.makedirs(folder_working_path, exist_ok=True)

image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]

step = 30
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.4
font_thickness = 1
min_distance = 15  

for filename in image_files:
    img_path = os.path.join(folder_path, filename)
    img = cv2.imread(img_path)

    if img is None:
        print('[INFO] Fail to upload')
        continue

    img_copy = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 30, 120)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    contour_dot = np.ones_like(img) * 255
    point_index = 1

    placed_points = []

    for contour in contours:
        for i in range(0, len(contour), step):
            x, y = contour[i][0]

            too_close = False
            for px, py in placed_points:
                if math.dist((x, y), (px, py)) < min_distance:
                    too_close = True
                    break

            cv2.circle(contour_dot, (x, y), 2, (0, 0, 0), -1)

            if too_close:
                continue  

            (text_w, text_h), _ = cv2.getTextSize(str(point_index), font, font_scale, font_thickness)
            text_x = x - text_w // 2
            text_y = y - 5  

            cv2.putText(contour_dot, str(point_index), (text_x, text_y), font, font_scale, (0, 0, 0), font_thickness, cv2.LINE_AA)

            placed_points.append((x, y))

            point_index += 1

    cv2.imshow('Dot Numbered Contours', contour_dot)
    cv2.imshow('original', img_copy)
    cv2.waitKey(0)

    save_path = os.path.join(folder_working_path, f"contour_dot_numbered_{filename}")
    cv2.imwrite(save_path, contour_dot)
    print(f"[INFO] {save_path} saved successfully!")

cv2.destroyAllWindows()
