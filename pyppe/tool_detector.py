
'''
object detection/tracking test
'''
import cv2

camera = cv2.VideoCapture(0)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1600)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 1200)
camera.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0)
camera.set(cv2.CAP_PROP_AUTO_WB, 0)
#tracker = cv2.legacy.TrackerMOSSE_create()
#tracker = cv2.legacy.TrackerCSRT_create()
tracker = cv2.legacy.TrackerMedianFlow_create()
success, img = camera.read()
bbox = cv2.selectROI("raw", img, False)
tracker.init(img, bbox)

def drawBox(raw, bbox):
    x,y,w,h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    cv2.rectangle(raw, (x,y), ((x+w), (y+h)), (255,0,0), 3, 1)

idx = 1
while True:
    timer = cv2.getTickCount()
    success, raw = camera.read()

    # success, bbox = tracker.update(raw)
    # if success:
    #     drawBox(raw, bbox)
    # else:
    #     cv2.putText(raw, "Lost", (75,75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)


    # fps = cv2.getTickFrequency()/(cv2.getTickCount()-timer)
    # cv2.putText(raw, str(fps), (75,50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
    cv2.imshow("camera", raw)

    key = cv2.waitKey(1)
    if key == 27:
        break
    elif key == ord('s') or key == ord('S'):
        cv2.imwrite("{}.jpg".format(idx), raw)
        print("captured : {}".format(idx))
        idx += 1

cv2.destroyAllWindows()