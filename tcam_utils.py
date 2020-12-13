import signal
import time 
import cv2
import TIS

import numpy as np
import affvisionpy as af

from face_listener import FaceListener as ImageListener
from object_listener import ObjectListener as ObjectListener
from occupant_listener import OccupantListener as OccupantListener
from body_listener import BodyListener as BodyListener

from display_metrics import (draw_affectiva_logo, check_bounding_box_outside, draw_bounding_box, draw_metrics,
                             draw_bodies, draw_objects, draw_and_calculate_3d_pose, draw_gaze_region, get_bounding_box_points, display_measurements,
                             display_left_metric, display_drowsiness, display_expression)

# TODO- fix this 
OBJECT_CALLBACK_INTERVAL = 500
OCCUPANT_CALLBACK_INTERVAL = 500
BODY_CALLBACK_INTERVAL = 500


class KillSignalHandler():
  killer = False
  def __init__(self):
    signal.signal(signal.SIGINT, self.exit_gracefully)
    signal.signal(signal.SIGTERM, self.exit_gracefully)

  def exit_gracefully(self, signum):
    self.killer = True


def start_detector(detector, features, listener):
    detector.enable_features(features)
    detector.set_image_listener(listener)
    detector.start()

def get_tcam_frame(tis, framerate):
    if tis.Snap_image(1/framerate) is True:
        # np array returned from tis.Get_image() is immutable
        frame = np.array(tis.Get_image())
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        return frame
    else:
        print("INFO: Snap_image failed")
        return None

def create_afframe(frame, start_time):
    height = frame.shape[0]
    width = frame.shape[1]
    curr_timestamp = (time.time() - start_time) * 1000.0
    afframe = af.Frame(width, height, frame, af.Frame.ColorFormat.bgr, int(curr_timestamp))
    return afframe

def get_face_input_results(listener, frame):
    listener.mutex.acquire()

    faces = listener.faces.copy()
    expressions_dict = listener.expressions_dict.copy()
    emotions_dict = listener.emotions_dict.copy()
    bounding_box_dict = listener.bounding_box_dict.copy()

    listener.mutex.release()

    listener_metrics = {
        "expressions": expressions_dict,
        "emotions": emotions_dict,
        "bounding_box": bounding_box_dict
    }

    height = frame.shape[0]
    width = frame.shape[1]
    if len(faces) > 0 and not check_bounding_box_outside(width, height, bounding_box_dict):
        draw_bounding_box(frame, listener_metrics, True)
        draw_metrics(frame, listener_metrics, {})

def get_3d_pose_input_results(listener, frame, camera_matrix, camera_type, dist_coefficients):
    listener.mutex.acquire()
    face_landmark_points_dict = listener.face_landmark_points_dict.copy()
    listener.mutex.release()

    listener_metrics = {
        "face_landmark_pts": face_landmark_points_dict
    }

    if (len(face_landmark_points_dict) > 0):
        draw_and_calculate_3d_pose(frame, camera_matrix, camera_type, dist_coefficients, listener_metrics)


def get_face_bbox_input_results(face_listener, frame):
    face_listener.mutex.acquire()

    faces = face_listener.faces.copy()
    bounding_box_dict = face_listener.bounding_box_dict.copy()

    face_listener.mutex.release()

    listener_metrics = {
        "bounding_box": bounding_box_dict
    }

    height = frame.shape[0]
    width = frame.shape[1]
    if len(faces) > 0 and not check_bounding_box_outside(width, height, bounding_box_dict):
        draw_bounding_box(frame, listener_metrics, False)


def tcam_process_face_input(detector, tis, start_time, output_file, out, logo, args, camera_matrix, dist_coefficients, camera_type="fisheye"):
    features = {af.Feature.expressions, af.Feature.emotions}
    listener = ImageListener()
    start_detector(detector, features, listener)

    kill_signal_handler = KillSignalHandler()
    while not kill_signal_handler.killer:
        frame = get_tcam_frame(tis, 30)
        if frame is not None:
            afframe = create_afframe(frame, start_time)
            try:
                detector.process(afframe)
            except Exception as exp:
                print(exp)

            if not args.no_draw:
                draw_affectiva_logo(frame, logo, frame.shape[1], frame.shape[0])
                get_face_input_results(listener, frame)
                get_3d_pose_input_results(listener, frame, camera_matrix, camera_type, dist_coefficients)
                cv2.imshow('Processed Frame', frame)

            if output_file is not None:
                out.write(frame)

            if cv2.waitKey(1) == 27:
                kill_signal_handler.exit_gracefully(signal.SIGTERM)

    print("Gracefully killing tcam stuff")
    detector.stop()


def get_object_input_results(object_listener, frame):
    object_listener.mutex.acquire()
    objects = object_listener.objects.copy()
    bounding_box_dict = object_listener.bounding_box.copy()
    type_dict = object_listener.type.copy()
    object_listener.mutex.release()

    listener_metrics = {
        "bounding_box": bounding_box_dict,
        "object_type": type_dict
    }

    if len(objects) > 0 and not check_bounding_box_outside(frame.shape[1], frame.shape[0], listener_metrics["bounding_box"]):
        draw_objects(frame, listener_metrics)

def get_body_input_results(body_listener, frame):
    body_listener.mutex.acquire()
    bodies = body_listener.bodies.copy()
    body_points_dict = body_listener.bodyPoints.copy()
    body_listener.mutex.release()

    listener_metrics = {
        "body_points": body_points_dict
    }

    if len(bodies) > 0:
        draw_bodies(frame, listener_metrics)


def tcam_process_object_input(detector, tis, start_time, output_file, out, logo, args):
    features = {af.Feature.phones, af.Feature.child_seats, af.Feature.bodies}
    detector.enable_features(features)

    # callback interval
    object_listener = ObjectListener(OBJECT_CALLBACK_INTERVAL)
    detector.set_object_listener(object_listener)

    # callback interval for body
    body_listener = BodyListener(BODY_CALLBACK_INTERVAL)
    detector.set_body_listener(body_listener)

    detector.start()

    kill_signal_handler = KillSignalHandler()
    while not kill_signal_handler.killer:
        frame = get_tcam_frame(tis, 30)
        if frame is not None:
            afframe = create_afframe(frame, start_time)
            try:
                detector.process(afframe)
            except Exception as exp:
                print(exp)

            if not args.no_draw:
                draw_affectiva_logo(frame, logo, frame.shape[1], frame.shape[0])
                get_object_input_results(object_listener, frame)
                get_body_input_results(body_listener, frame)
                cv2.imshow('Processed Frame', frame)

            if output_file is not None:
                out.write(frame)

            if cv2.waitKey(1) == 27:
                kill_signal_handler.exit_gracefully(signal.SIGTERM)

    print("Gracefully killing tcam stuff")
    detector.stop()


def tcam_process_occupant_bkp_input(detector, tis, start_time, output_file, out, logo, args):
    features = {af.Feature.faces, af.Feature.bodies}
    detector.enable_features(features)

    face_listener = ImageListener()
    detector.set_image_listener(face_listener)

    # callback interval for body
    body_listener = BodyListener(BODY_CALLBACK_INTERVAL)
    detector.set_body_listener(body_listener)

    detector.start()

    kill_signal_handler = KillSignalHandler()
    while not kill_signal_handler.killer:
        frame = get_tcam_frame(tis, 30)
        if frame is not None:
            afframe = create_afframe(frame, start_time)
            try:
                detector.process(afframe)
            except Exception as exp:
                print(exp)

            if not args.no_draw:
                draw_affectiva_logo(frame, logo, frame.shape[1], frame.shape[0])
                get_face_bbox_input_results(face_listener, frame)
                get_body_input_results(body_listener, frame)
                cv2.imshow('Processed Frame', frame)

            if output_file is not None:
                out.write(frame)

            if cv2.waitKey(1) == 27:
                kill_signal_handler.exit_gracefully(signal.SIGTERM)

    print("Gracefully killing tcam stuff")
    detector.stop()


def get_gaze_input_results(face_listener, frame):
    face_listener.mutex.acquire()
    gaze_metrics = face_listener.gaze_metric_dict.copy()
    face_listener.mutex.release()

    listener_metrics = {
        "gaze_region": gaze_metrics
    }

    if len(gaze_metrics):
        metric = next(iter(gaze_metrics.values()))
        draw_gaze_region(frame, metric)

def get_gaze_bbox_input_results(face_listener, frame):
    face_listener.mutex.acquire()

    faces = face_listener.faces.copy()
    bounding_box_dict = face_listener.bounding_box_dict.copy()
    measurements_dict = face_listener.measurements_dict.copy()
    expressions_dict = face_listener.expressions_dict.copy()
    glasses_dict = face_listener.glasses_dict.copy()

    face_listener.mutex.release()

    listener_metrics = {
        "bounding_box": bounding_box_dict,
        "measurements": measurements_dict,
        "glasses": glasses_dict,
        "expressions": expressions_dict
    }

    height = frame.shape[0]
    width = frame.shape[1]
    if len(faces) > 0 and not check_bounding_box_outside(width, height, bounding_box_dict):
        draw_bounding_box(frame, listener_metrics, False)
        for fid in faces:
            upper_left_x, upper_left_y, lower_right_x, lower_right_y = get_bounding_box_points(fid, listener_metrics["bounding_box"])
            box_width = lower_right_x - upper_left_x
            upper_right_x = upper_left_x + box_width
            upper_right_y = upper_left_y

            for key, val in measurements_dict[fid].items():
                display_measurements(key.name, val, upper_left_y, frame, upper_left_x)
                upper_left_y += 25

            display_left_metric("eye_closure", expressions_dict[fid][af.Expression.eye_closure], upper_left_x, upper_left_y, frame)
            upper_left_y += 25

            display_left_metric("glasses", glasses_dict[fid], upper_left_x, upper_left_y, frame)
            upper_left_y += 25


def tcam_process_gaze_input(detector, tis, start_time, output_file, out, logo, args, camera_matrix, dist_coefficients, camera_type="fisheye"):
    features = {af.Feature.faces, af.Feature.gaze, af.Feature.appearances, af.Feature.expressions}
    detector.enable_features(features)

    listener = ImageListener()
    detector.set_image_listener(listener)

    detector.start()

    kill_signal_handler = KillSignalHandler()
    while not kill_signal_handler.killer:
        frame = get_tcam_frame(tis, 30)
        if frame is not None:
            afframe = create_afframe(frame, start_time)
            try:
                detector.process(afframe)
            except Exception as exp:
                print(exp)

            if not args.no_draw:
                draw_affectiva_logo(frame, logo, frame.shape[1], frame.shape[0])
                get_gaze_input_results(listener, frame)
                get_gaze_bbox_input_results(listener, frame)
                get_3d_pose_input_results(listener, frame, camera_matrix, camera_type, dist_coefficients)
                cv2.imshow('Processed Frame', frame)

            if output_file is not None:
                out.write(frame)

            if cv2.waitKey(1) == 27:
                kill_signal_handler.exit_gracefully(signal.SIGTERM)

    print("Gracefully killing tcam stuff")
    detector.stop()


def get_drowsiness_input_results(face_listener, frame):
    face_listener.mutex.acquire()

    faces = face_listener.faces.copy()
    bounding_box_dict = face_listener.bounding_box_dict.copy()
    expressions_dict = face_listener.expressions_dict.copy()
    glasses_dict = face_listener.glasses_dict.copy()
    drowsiness_dict = face_listener.drowsiness_dict.copy()

    face_listener.mutex.release()

    listener_metrics = {
        "bounding_box": bounding_box_dict,
        "glasses": glasses_dict,
        "expressions": expressions_dict
    }

    height = frame.shape[0]
    width = frame.shape[1]
    if len(faces) > 0 and not check_bounding_box_outside(width, height, bounding_box_dict):
        draw_bounding_box(frame, listener_metrics, False)
        for fid in faces:
            upper_left_x, upper_left_y, lower_right_x, lower_right_y = get_bounding_box_points(fid, listener_metrics["bounding_box"])
            box_width = lower_right_x - upper_left_x
            upper_right_x = upper_left_x + box_width
            upper_right_y = upper_left_y

            display_drowsiness(frame, drowsiness_dict[fid], upper_left_x, upper_left_y)
            upper_left_y += 25
            display_left_metric("drowsiness confidence", drowsiness_dict[fid].confidence, upper_left_x, upper_left_y, frame)
            upper_left_y += 25

            display_expression("eye_closure", expressions_dict[fid][af.Expression.eye_closure], upper_right_x, upper_right_y, frame)
            upper_right_y += 25
            display_expression("yawn", expressions_dict[fid][af.Expression.yawn], upper_right_x, upper_right_y, frame)
            upper_right_y += 25
            display_expression("blink_rate", expressions_dict[fid][af.Expression.blink_rate], upper_right_x, upper_right_y, frame)
            upper_right_y += 25
            display_expression("glasses", glasses_dict[fid], upper_right_x, upper_right_y, frame)
            upper_right_y += 25


def tcam_process_drowsiness_input(detector, tis, start_time, output_file, out, logo, args, camera_matrix, dist_coefficients, camera_type="fisheye"):
    features = {af.Feature.faces, af.Feature.appearances, af.Feature.expressions, af.Feature.drowsiness}
    detector.enable_features(features)

    listener = ImageListener()
    detector.set_image_listener(listener)

    detector.start()

    kill_signal_handler = KillSignalHandler()
    while not kill_signal_handler.killer:
        frame = get_tcam_frame(tis, 30)
        if frame is not None:
            afframe = create_afframe(frame, start_time)
            try:
                detector.process(afframe)
            except Exception as exp:
                print(exp)

            if not args.no_draw:
                draw_affectiva_logo(frame, logo, frame.shape[1], frame.shape[0])
                get_drowsiness_input_results(listener, frame)
                get_3d_pose_input_results(listener, frame, camera_matrix, camera_type, dist_coefficients)
                cv2.imshow('Processed Frame', frame)

            if output_file is not None:
                out.write(frame)

            if cv2.waitKey(1) == 27:
                kill_signal_handler.exit_gracefully(signal.SIGTERM)

    print("Gracefully killing tcam stuff")
    detector.stop()

