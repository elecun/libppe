'''
Image file extractor from video file
'''

import cv2
import argparse
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', nargs='?', required=True, help="input video file to extract images")
    args = parser.parse_args()

    if args.video is not None:
        print(args.video)
        _path, _ext = os.path.splitext(args.video)
        frame_count = 0
        if _ext == '.avi':
            _video = cv2.VideoCapture(args.video)
            _width  = int(_video.get(cv2.CAP_PROP_FRAME_WIDTH))
            _height = int(_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
            _fps = _video.get(cv2.CAP_PROP_FPS)
            _frames = int(_video.get(cv2.CAP_PROP_FRAME_COUNT))
            print("> Video source info. : ({},{}@{}), {} frames".format(_width, _height, _fps, _frames))

            if _video.isOpened():
                while True:
                    ret, frame = _video.read()
                    if ret == True:
                        target_filename = _path+"_"+str(frame_count)+".png"
                        cv2.imwrite(target_filename, frame)
                        frame_count+=1
                    else:
                        break
        print("finished")