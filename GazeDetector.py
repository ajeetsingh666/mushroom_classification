import os
import cv2
import math
import mediapipe as mp
# from iris_folder import filter_fsla
 
import sys

# retina_path = os.path.abspath('/home/ajeet/codework/backend_vir_env/visiontasks/RetinaNet_Face_Verification')
# sys.path.append(retina_path)

# from retinanet_test import ajeesing_test_method

import numpy as np

class GazeDetector:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(refine_landmarks=True)
        self.blink_detector = BlinkDetector()

    def euclidean_distance(self, p1, p2, w, h):
        x1, y1 = int(p1.x * w), int(p1.y * h)
        x2, y2 = int(p2.x * w), int(p2.y * h)
        return math.hypot(x2 - x1, y2 - y1)

    def get_face_bbox_area(self, landmarks, w, h):
        xs = [int(pt.x * w) for pt in landmarks]
        ys = [int(pt.y * h) for pt in landmarks]
        return (max(xs) - min(xs)) * (max(ys) - min(ys))

    def get_eye_ratio(self, outer, inner, pupil):
        return abs(pupil.x - outer.x) / abs(inner.x - outer.x)

    def get_gaze_direction(self, image, landmarks, img_path):
        # h, w = image.shape[:2]
        # rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # results = self.face_mesh.process(rgb)


        left_ratio = self.get_eye_ratio(landmarks[33], landmarks[133], landmarks[468])
        # # right_ratio = self.get_eye_ratio(landmarks[362], landmarks[263], landmarks[473])
        right_ratio = self.get_eye_ratio(landmarks[263], landmarks[362], landmarks[473])
        avg_ratio = (left_ratio + right_ratio) / 2.0

        ratio_min = min(left_ratio, right_ratio)
        ratio_max = max(left_ratio, right_ratio)

        # total_left_white, total_right_white = self.white_pixel_detector.get_white_pixel_gaze(image, landmarks, img_path)

        # white_pixel_ratio = 0
        # if total_right_white >= 0:
        #     white_pixel_ratio = total_left_white/(total_right_white+0.00001)

        # pixel_based_gaze = "Looking RIGHT"
        # if white_pixel_ratio < 0.5:
        #     pixel_based_gaze = "Looking RIGHT"
        # elif white_pixel_ratio > 1.5:
        #     pixel_based_gaze = "Looking LEFT"


        ratio_based_gaze = "GAZE CENTER"
        final_gaze = "GAZE CENTER"
        Not_in_ratio_limit = False
        # if ratio_max <= 2 * ratio_min:
        #     if avg_ratio < 0.38:
        #         ratio_based_gaze = "Looking RIGHT"
        #     elif avg_ratio > 0.62:
        #         ratio_based_gaze = "Looking LEFT"
        # else:
        #     print("Not_in_1.5", img_path)
        #     Not_in_ratio_limit = True
        #     ratio_based_gaze = "Not_in_ratio_limit"

        # final_gaze = ratio_based_gaze
        

        if left_ratio < 0.40 and right_ratio > 0.55:
            ratio_based_gaze = "GAZE RIGHT"
        elif left_ratio > 0.55 and right_ratio < 0.40:
            ratio_based_gaze = "GAZE LEFT"
        else:
            ratio_based_gaze = "GAZE CENTER"

        final_gaze = ratio_based_gaze

        # if ratio_based_gaze == pixel_based_gaze:
        #     final_gaze = ratio_based_gaze

        # debug_info = f"LRatio: {left_ratio:.2f}, RRatio: {right_ratio:.2f}, Avg: {avg_ratio:.2f}, " \
        #                 f"WhiteL: {total_left_white}, WhiteR: {total_right_white}, Gaze: {final_gaze}"
        debug_info = f"LRatio: {left_ratio:.2f}, RRatio: {right_ratio:.2f}, Avg: {avg_ratio:.2f}, Gaze: {final_gaze}"
        print(debug_info)

        # return final_gaze, left_ratio, right_ratio, avg_ratio, total_left_white, total_right_white, Not_in_ratio_limit

        return final_gaze, left_ratio, right_ratio, avg_ratio, Not_in_ratio_limit

    def detect_gaze(self, frames_list, merged_incident_list, aligned_faces_path):
        """
        Accepts a list of frames (BGR images) and returns a list of booleans.
        True if gaze is LEFT or RIGHT, False if CENTER or gaze couldn't be determined.
        """
        gaze_bools = []

        for idx, _ in enumerate(frames_list):
            if merged_incident_list[idx]:
                gaze_bools.append(False)
                continue

            aligned_face = os.path.join(aligned_faces_path, os.path.basename(frames_list[idx]))
            
            if os.path.exists(aligned_face):
                h, w = frames_list[idx].shape[:2]
                rgb = cv2.cvtColor(frames_list[idx], cv2.COLOR_BGR2RGB)
                results = self.face_mesh.process(rgb)

                if results.multi_face_landmarks:
                    landmarks = results.multi_face_landmarks[0].landmark

                blink_result = self.blink_detector.is_blinking(landmarks, w, h)
                blinking, avg_ear, (left_ear, right_ear) = blink_result
                if blinking is not None and blinking:
                    gaze_bools.append(False)
                    continue

                gaze, left_ratio, right_ratio, avg_ratio, not_in_ratio_limit = self.get_gaze_direction(frame, landmarks, img_path=None)
                gaze_bools.append(gaze in ["GAZE LEFT", "GAZE RIGHT"])

                debug = True
                if debug:
                    output_base = "/home/ajeet/codework/testing_gaze"
                    output_folders = {
                    "GAZE LEFT": [os.path.join(output_base, "left"), os.path.join(output_base, "face_left")],
                    "GAZE RIGHT": [os.path.join(output_base, "right"), os.path.join(output_base, "face_right")],
                    "GAZE CENTER": [os.path.join(output_base, "center"), os.path.join(output_base, "face_center")],
                    "Blinking": [os.path.join(output_base, "blinking")],
                    }
                    for folder_list in output_folders.values():
                        for folder in folder_list:
                            os.makedirs(folder, exist_ok=True)

                    actual_image_frame= cv2.imread(frames_list[idx])
                    cv2.putText(actual_image_frame, f"Left: {left_ratio:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                    cv2.putText(actual_image_frame, f"Right: {right_ratio:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                    cv2.putText(actual_image_frame, f"Avg: {avg_ratio:.2f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

                    # cv2.putText(actual_image_frame, f"LW: {total_left_white}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    # cv2.putText(actual_image_frame, f"RW: {total_right_white}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(actual_image_frame, f"{gaze}", (10, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

                    save_path = output_folders.get(gaze, output_folders[gaze])
                    file_name= os.path.basename(frames_list[idx])
                    output_path = os.path.join(save_path[0], f"{file_name}")
                    cv2.imwrite(output_path, actual_image_frame)

                else:
                    gaze_bools.append(False)
            else:
                gaze_bools.append(False)

        return gaze_bools




if __name__ == '__main__':
    pass
