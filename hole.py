import cv2, numpy as np


def proccess_holes(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return (0, 0)
    hier = hierarchy[0]
    candidates = [i for i, h in enumerate(hier) if h[3] < 0]
    if not candidates:
        return (0, 0)
    areas = [cv2.contourArea(contours[i]) for i in candidates]
    bg_idx = candidates[int(np.argmax(areas))]
    shape_idxs = [i for i, h in enumerate(hier) if h[3] == bg_idx]
    hole_counts = {}
    for idx in shape_idxs:
        cnt = 0
        child = hier[idx][2]
        while child != -1:
            cnt += 1
            child = hier[child][0]
        hole_counts[idx] = cnt
    if not hole_counts:
        return (0, 0)
    max_idx = max(hole_counts, key=hole_counts.get)
    M = cv2.moments(contours[max_idx])
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
    else:
        cx, cy = 0, 0
    return cx, cy

path = "ddd6d8add515201e0ed0313d7d21aef78cc96d026d16526d2d6fedb61f34165b.jpg"
x, y = proccess_holes(path)
img = cv2.imread(path)
cv2.circle(img, (x, y), 10, (0, 0, 255), -1)
cv2.putText(img, f"({x},{y})", (x + 15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
cv2.imwrite("output_with_point.png", img)
