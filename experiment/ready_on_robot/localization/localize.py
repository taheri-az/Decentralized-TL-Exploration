import cv2
import json
import numpy as np
from math import atan2, pi
from pathlib import Path

with open(str(Path(__file__).absolute().parent / "camera.json"), "r") as json_file:
    camera_data = json.load(json_file)
dist = np.array(camera_data["dist"])
mtx = np.array(camera_data["mtx"])
# These are calculated using an aruco market with side 18cms
origin_rvec = np.array(camera_data["origin_rvec"])
origin_R = cv2.Rodrigues(origin_rvec)[0]
origin_tvec = np.array(camera_data["origin_tvec"]).reshape(3, 1)

landmarks = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]])


def convert_coord(x, rvec, tvec):
    R = cv2.Rodrigues(rvec)[0]
    camera_frame_coord = R @ x.reshape(3, 1) + tvec
    return np.linalg.inv(origin_R) @ (camera_frame_coord - origin_tvec)


dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
parameters = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(dictionary, parameters)
h, w = 1200, 1920
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
mapx, mapy = cv2.initUndistortRectifyMap(
    mtx, dist, None, newcameramtx, (w, h), cv2.CV_32FC1
)

def localize_fast(frame):
    undistorted = cv2.remap(frame, mapx, mapy, cv2.INTER_LINEAR)
    # undistorted = undistorted[y:y+h, x:x+w]
    cv2.imwrite("frame.jpg", frame)
    cv2.imwrite("und.jpg", undistorted)

    (corners, ids, rejected) = detector.detectMarkers(undistorted)
    if len(corners) == 0:
        # print("Did not found corners")
        return None, None, None
        # exit()
    ids = list(map(lambda x: int(x), ids.flatten()))

    locations = []
    headings = []
    for i in range(len(ids)):
        aruco_marker_size = 0.12
        nada, rvec, tvec = cv2.solvePnP(
            landmarks * aruco_marker_size, corners[i], mtx, dist
        )
        # print(rvec, tvec)
        # cv2.drawFrameAxes(frame, mtx, dist, rvec, tvec, 0.2)

        x0 = convert_coord(landmarks[0], rvec, tvec)
        x1 = convert_coord(landmarks[3], rvec, tvec)
        v = (x0 - x1)[:2].flatten()
        theta = atan2(v[1], v[0]) / pi * 180
        if theta < 0:
            theta += 360

        # print("------", i)
        # print(x0)
        locations.append(x0.flatten().tolist()[:2])
        headings.append(theta)

    return ids, locations, headings


def localize(frame):
    # ----- undistort: Do I need this?
    # h, w = frame.shape[:2]
    # newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    # mapx, mapy = cv2.initUndistortRectifyMap(
    # 	mtx, dist, None, newcameramtx, (w, h), cv2.CV_32FC1
    # )
    # x, y, w, h = roi
    # undistorted = cv2.remap(frame, mapx, mapy, cv2.INTER_LINEAR)
    # undistorted = undistorted[y:y+h, x:x+w]

    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(dictionary, parameters)
    (corners, ids, rejected) = detector.detectMarkers(frame)

    drawaxes = True
    if len(corners) == 0:
        print("Did not found corners")
        exit()

    locations = []

    if drawaxes:
        for i in range(len(ids)):
            aruco_marker_size = 0.12  # in meters
            nada, rvec, tvec = cv2.solvePnP(
                landmarks * aruco_marker_size, corners[i], mtx, dist
            )
            cv2.drawFrameAxes(frame, mtx, dist, rvec, tvec, 0.2)

            R = cv2.Rodrigues(rvec)[0]
            camera_frame_coord = R @ landmarks[0].reshape(3, 1) + tvec

            location = np.linalg.inv(origin_R) @ (camera_frame_coord - origin_tvec)
            locations.append(location)

    ids = ids.flatten()
    cv2.aruco.drawDetectedMarkers(frame, corners, ids)

    cv2.imshow("frame", frame)
    key = cv2.waitKey()


if __name__ == "__main__":
    # Use 4 for capturing USB C camera (Better one) / Use 1 for webcam
    vid = cv2.VideoCapture(4)
    # vid = cv2.VideoCapture(1, cv2.CAP_V4L2)
    # vid = cv2.VideoCapture(
    #     "v4l2src device=/dev/video1 ! video/x-raw,width=1920,height=1200 ! videoconvert ! appsink",
    #     cv2.CAP_GSTREAMER
    # )



    width = vid.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    height = vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 1200)
    fps = vid.get(cv2.CAP_PROP_FPS)
    # print(width, height, fps)


    while True:
        ret, frame = vid.read()

        # Display the resulting frame
        # print("Showing img")
        cv2.imshow("frame", frame)

        key = cv2.waitKey(1)
        if key == ord("q"):
            break
        if key == ord("c"):
            continue
        if key == ord("l"):
            # localize(frame)
            cv2.imwrite(f"a.jpg", frame)
            print(localize_fast(frame))


    vid.release()
    cv2.destroyAllWindows()


# sudo modprobe -r uvcvideo && sleep 0.3 && sudo modprobe uvcvideo