import cv2
import glob
import numpy as np
from glob import glob

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

chessboard_x = 9
chessboard_y = 6
chessboard_cell_size = 0.0226
landmarks = np.zeros((chessboard_x * chessboard_y, 3), np.float32)
landmarks[:, :2] = np.mgrid[0:chessboard_x, 0:chessboard_y].T.reshape(-1, 2) * chessboard_cell_size

all_landmarks = []  # 3d point in real world space
projected_points = []  # 2d points in image plane.


def calibrate(frames):
    # fname = ""
    # img = cv.imread(fname)
    # cv.imshow('img', img)
    # cv.waitKey(0)

    for img in frames:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(
            gray, (chessboard_x, chessboard_y), None
        )
        print("corners: ", not corners is None)
        # print("corners: ", corners)

        # If I could find chess board pattern
        if ret == True:
            all_landmarks.append(landmarks)
            accurate_corners = cv2.cornerSubPix(
                gray, corners, (3, 3), (-1, -1), criteria
            )
            # accurate_corners = corners
            projected_points.append(accurate_corners)
            cv2.drawChessboardCorners(
                img, (chessboard_x, chessboard_y), accurate_corners, ret
            )
            cv2.imshow("window", img)
            cv2.waitKey()

        else:
            print("Couldn't find corners, skip")

    _, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        all_landmarks, projected_points, gray.shape[::-1], None, None
    )
    print("Calibration result, K: ", K)
    for i in range(3):
        print(",".join(map(str, K[i])))
    print("Calibration result, dist: ", dist)
    print(",".join(map(str, dist)))
    # print("Calibration result, rvecs: ", rvecs)
    # print("Calibration result, tvecs: ", tvecs)

    mean_error = 0
    all_errors = []
    for i in range(len(projected_points)):
        imgpoints2, _ = cv2.projectPoints(landmarks, rvecs[i], tvecs[i], K, dist)
        error = cv2.norm(projected_points[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        mean_error += error
        all_errors.append(error)

    print("total error: {}".format(mean_error / len(projected_points)))

    cv2.destroyAllWindows()


if __name__ == "__main__":
    frames = []
    for i, f in enumerate(glob("experiment/localization/saved_chessboard/*.png")):
        print("f: ", f)
        frame = cv2.imread(f)
        frames.append(frame)
    calibrate(frames)
