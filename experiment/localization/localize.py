import cv2
import json
import numpy as np
from math import atan2, pi
from pathlib import Path

with open(str(Path(__file__).absolute().parent / "camera.json"), "r") as json_file:
    camera_data = json.load(json_file)
dist = np.array(camera_data["dist"])
mtx = np.array(camera_data["mtx"])
origin_rvec = np.array(camera_data["origin_rvec"])
origin_R = cv2.Rodrigues(origin_rvec)[0]
origin_tvec = np.array(camera_data["origin_tvec"]).reshape(3, 1)
origin_R_inv = np.linalg.inv(origin_R)

landmarks = np.array([
    [-1,  1, 0],
    [ 1,  1, 0],
    [ 1, -1, 0],
    [-1, -1, 0]
], dtype=np.float32)

ORIGIN_MARKER_SIZE = 0.1778
ROBOT_MARKER_SIZE = 0.12

dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
parameters = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(dictionary, parameters)
h, w = 1200, 1920
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
mapx, mapy = cv2.initUndistortRectifyMap(
    mtx, dist, None, newcameramtx, (w, h), cv2.CV_32FC1
)


def convert_coord(x, rvec, tvec):
    R = cv2.Rodrigues(rvec)[0]
    cam_pt = R @ x.reshape(3, 1) + tvec
    origin_pt = origin_R_inv @ (cam_pt - origin_tvec)
    return origin_pt[:2]

def get_heading(rvec):
    R = cv2.Rodrigues(rvec)[0]
    marker_y_cam = R[:, 1]  # try Y-axis instead
    marker_y_origin = (origin_R_inv @ marker_y_cam).flatten()
    theta = atan2(marker_y_origin[1], marker_y_origin[0]) * 180 / pi
    if theta < 0:
        theta += 360
    return theta


visualize = False
def localize_fast(frame):
    (corners, ids, rejected) = detector.detectMarkers(frame)
    if len(corners) == 0:
        return None, None, None
    ids = list(map(lambda x: int(x), ids.flatten()))

    locations = []
    headings = []
    for i in range(len(ids)):
        # marker_size = ORIGIN_MARKER_SIZE if ids[i] == 0 else ROBOT_MARKER_SIZE
        # marker_size = ORIGIN_MARKER_SIZE 
        marker_size = ROBOT_MARKER_SIZE
        obj_pts = landmarks * marker_size / 2

        n_sol, rvecs, tvecs, reproj = cv2.solvePnPGeneric(
            obj_pts, corners[i], mtx, dist,
            flags=cv2.SOLVEPNP_IPPE_SQUARE
        )
        best = min(range(n_sol), key=lambda s: cv2.Rodrigues(rvecs[s])[0][2, 2])
        rvec, tvec = rvecs[best], tvecs[best]

        if visualize:
            cv2.drawFrameAxes(frame, mtx, dist, rvec, tvec, 0.2)

        # Position (center of marker)
        center = convert_coord(np.array([0., 0., 0.]), rvec, tvec)
        pos = center.flatten().tolist()
        pos_standard = [pos[0], pos[1]]


        theta = get_heading(rvec)

        # draw heading
        center_3d = np.array([[0., 0., 0.]])
        forward_3d = np.array([[marker_size, 0., 0.]])
        center_2d, _ = cv2.projectPoints(center_3d, rvec, tvec, mtx, dist)
        forward_2d, _ = cv2.projectPoints(forward_3d, rvec, tvec, mtx, dist)
        if visualize:
            c = tuple(center_2d[0].ravel().astype(int))
            f = tuple(forward_2d[0].ravel().astype(int))

            cv2.arrowedLine(frame, c, f, (0, 255, 255), 3, tipLength=0.3)
            cv2.putText(frame, f"ID:{ids[i]} h:{theta:.1f}",
                        (c[0] + 10, c[1] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 255, 255), 2)

        locations.append(pos_standard)
        headings.append(theta)

    # cv2.aruco.drawDetectedMarkers(frame, corners, np.array(ids))
    # print(ids, locations, headings)
    # cv2.imshow("frame", frame)
    # cv2.waitKey()

    return ids, locations, headings


if __name__ == "__main__":
    vid = cv2.VideoCapture(4, cv2.CAP_V4L2)
    vid.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 1200)
    vid.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    vid.set(cv2.CAP_PROP_FPS, 10)
    print(f"FPS: {vid.get(cv2.CAP_PROP_FPS)}")

    cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("frame", 640, 480)
    while True:
        ret, frame = vid.read()
        cv2.imshow("frame", frame)

        key = cv2.waitKey(1)
        if key == ord("q"):
            break
        if key == ord("c"):
            continue
        if key == ord("l"):
            cv2.imwrite("a.jpg", frame)
            print(localize_fast(frame))

    vid.release()
    cv2.destroyAllWindows()