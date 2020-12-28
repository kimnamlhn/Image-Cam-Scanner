import cv2
import numpy as np

#ham tim duong vien
def mapp(h):
    h = h.reshape((4, 2))
    hnew = np.zeros((4, 2), dtype=np.float32)

    add = h.sum(1)
    hnew[0] = h[np.argmin(add)]
    hnew[2] = h[np.argmax(add)]

    diff = np.diff(h, axis=1)
    hnew[1] = h[np.argmin(diff)]
    hnew[3] = h[np.argmax(diff)]

    return hnew


#doc anh vao va resize anh
image = cv2.imread("img.jpg")
image = cv2.resize(image, (800, 600))
#show anh goc
cv2.imshow("Source Image", image)
#tao ban sao
orig = image.copy()

#chuyen anh sang gray scale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#blur
blurred = cv2.GaussianBlur(gray, (5, 5),
                           0)
#canny
edged = cv2.Canny(blurred, 30, 50)


# truy xuat cac duong vien
contours, hierarchy = cv2.findContours(edged, cv2.RETR_LIST,
                                       cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key=cv2.contourArea, reverse=True)

# Lap tim cac duong vien cua hinh anh
for c in contours:
    p = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * p, True)

    if len(approx) == 4:
        target = approx
        break
approx = mapp(target)  # tim di vien cua to giay

pts = np.float32([[0, 0], [800, 0], [800, 800], [0, 800]])  # cua so anh 800*800

op = cv2.getPerspectiveTransform(approx, pts)
dst = cv2.warpPerspective(orig, op, (800, 800))

#show anh sau xu ly
cv2.imshow("Destination Image", dst)

# man hinh cho de so sanh 2 anh src va des
cv2.waitKey(0)
cv2.destroyAllWindows()





