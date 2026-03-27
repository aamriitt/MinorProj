"""
app.py
Main Flask server with:
  • Live MJPEG video stream  (/video)
  • REST API routes           (/, /alerts, /voice, /clear)
  • WebSocket (SocketIO)      for real-time alert push
  • Background monitoring thread
"""

import os
import time
import threading
import cv2
from flask import Flask, Response, render_template, jsonify
from flask_socketio import SocketIO, emit

from motion_detection import MotionDetector
from inactivity_monitor import InactivityMonitor
from fall_detection import FallDetector
from voice_ai import VoiceAI
from notifier import Notifier
from db import Database

# ------------------------------------------------------------------ #
# Configuration – edit these for your environment                     #
# ------------------------------------------------------------------ #
CAMERA_INDEX = 0                  # 0 = default webcam
INACTIVITY_THRESHOLD = 30         # seconds before inactivity alert
SENDER_EMAIL = ""                 # your Gmail address
SENDER_PASSWORD = ""              # Gmail App Password
RECIPIENT_EMAIL = ""              # caregiver email


# ------------------------------------------------------------------ #
# App bootstrap                                                        #
# ------------------------------------------------------------------ #
app = Flask(__name__, template_folder="templates", static_folder="static")
app.config["SECRET_KEY"] = "elderly-monitor-secret"
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

db = Database()
notifier = Notifier(SENDER_EMAIL, SENDER_PASSWORD, RECIPIENT_EMAIL)
voice_ai = VoiceAI(model_size="tiny")

motion_detector = MotionDetector(sensitivity=25, min_area=500)
fall_detector = FallDetector()
inactivity_monitor = InactivityMonitor(threshold_seconds=INACTIVITY_THRESHOLD)

# Shared state visible to all threads
_camera = None
_camera_lock = threading.Lock()
_latest_frame = None
_frame_lock = threading.Lock()
_system_status = {
    "monitoring": False,
    "camera_ok": False,
    "inactive_duration": 0,
    "motion": False,
    "fall": False,
}


# ------------------------------------------------------------------ #
# Helper – push alert everywhere                                       #
# ------------------------------------------------------------------ #
def _fire_alert(message: str, alert_type: str = "general"):
    row_id = db.insert_alert(message, alert_type)
    payload = {
        "id": row_id,
        "type": alert_type,
        "message": message,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    socketio.emit("new_alert", payload)
    print(f"[ALERT] [{alert_type}] {message}")


# ------------------------------------------------------------------ #
# Inactivity callback                                                  #
# ------------------------------------------------------------------ #
def _on_inactivity(duration):
    msg = f"No movement detected for {duration:.0f} seconds."
    _fire_alert(msg, "inactivity")
    notifier.send_inactivity_alert(duration)
    voice_ai.speak("Alert! No movement detected. Are you okay?")


inactivity_monitor.on_alert = _on_inactivity


# ------------------------------------------------------------------ #
# Fall callback                                                        #
# ------------------------------------------------------------------ #
def _on_fall():
    msg = "Possible fall detected by the monitoring system."
    _fire_alert(msg, "fall")
    notifier.send_fall_alert()
    voice_ai.speak("Fall detected! Alerting your caregiver immediately.")


fall_detector.on_fall = _on_fall


# ------------------------------------------------------------------ #
# Background monitoring thread                                         #
# ------------------------------------------------------------------ #
def _monitoring_loop():
    global _camera, _latest_frame

    cap = cv2.VideoCapture(CAMERA_INDEX)
    with _camera_lock:
        _camera = cap

    if not cap.isOpened():
        _system_status["camera_ok"] = False
        print("[Monitor] ⚠ Camera not available – running in demo mode.")
        _demo_loop()
        return

    _system_status["camera_ok"] = True
    _system_status["monitoring"] = True
    inactivity_monitor.start()

    while _system_status["monitoring"]:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.05)
            continue

        frame, motion = motion_detector.detect(frame)
        frame, fall = fall_detector.analyse(frame)

        if motion:
            inactivity_monitor.update_motion()

        _system_status["motion"] = motion
        _system_status["fall"] = fall
        _system_status["inactive_duration"] = inactivity_monitor.get_inactive_duration()

        with _frame_lock:
            _latest_frame = frame.copy()

        time.sleep(0.03)   # ~30 fps

    cap.release()


def _demo_loop():
    """Runs when no camera is present; serves a placeholder frame."""
    _system_status["monitoring"] = True
    inactivity_monitor.start()

    placeholder = _make_placeholder()

    while _system_status["monitoring"]:
        _system_status["inactive_duration"] = inactivity_monitor.get_inactive_duration()
        with _frame_lock:
            _latest_frame = placeholder.copy()
        time.sleep(1)


def _make_placeholder():
    import numpy as np
    img = np.zeros((480, 640, 3), dtype="uint8")
    img[:] = (30, 30, 30)
    cv2.putText(img, "No Camera Found", (140, 220),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (80, 80, 200), 2)
    cv2.putText(img, "Running in demo mode", (130, 270),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (120, 120, 120), 1)
    return img


# Start monitoring thread on import
_monitor_thread = threading.Thread(target=_monitoring_loop, daemon=True)
_monitor_thread.start()


# ------------------------------------------------------------------ #
# MJPEG generator                                                      #
# ------------------------------------------------------------------ #
def _generate_frames():
    while True:
        with _frame_lock:
            frame = _latest_frame

        if frame is None:
            time.sleep(0.05)
            continue

        _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
        )
        time.sleep(0.04)


# ------------------------------------------------------------------ #
# Routes                                                               #
# ------------------------------------------------------------------ #

@app.route("/")
def dashboard():
    return render_template("dashboard.html")


@app.route("/video")
def video_feed():
    return Response(
        _generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/alerts")
def get_alerts():
    alerts = db.fetch_alerts(limit=50)
    return jsonify(alerts)


@app.route("/status")
def get_status():
    status = dict(_system_status)
    status["alert_count"] = db.count_alerts()
    return jsonify(status)


@app.route("/voice", methods=["POST"])
def trigger_voice():
    """Trigger a voice interaction in a background thread."""
    def _interact():
        text, response = voice_ai.interact()
        _fire_alert(f"Voice: \"{text}\" → \"{response}\"", "voice")

    threading.Thread(target=_interact, daemon=True).start()
    return jsonify({"status": "listening"})


@app.route("/clear", methods=["POST"])
def clear_alerts():
    db.clear_alerts()
    socketio.emit("alerts_cleared", {})
    return jsonify({"status": "cleared"})


@app.route("/test_alert", methods=["POST"])
def test_alert():
    """Convenience route to inject a test alert during demo."""
    _fire_alert("Test alert triggered from dashboard.", "test")
    return jsonify({"status": "ok"})


# ------------------------------------------------------------------ #
# SocketIO events                                                      #
# ------------------------------------------------------------------ #

@socketio.on("connect")
def on_connect():
    # Send current alert history on connect
    emit("alert_history", db.fetch_alerts(limit=50))
    emit("status_update", dict(_system_status))


@socketio.on("request_status")
def on_request_status():
    emit("status_update", dict(_system_status))


# ------------------------------------------------------------------ #
# Periodic status broadcast                                            #
# ------------------------------------------------------------------ #
def _status_broadcast():
    while True:
        time.sleep(3)
        socketio.emit("status_update", dict(_system_status))


threading.Thread(target=_status_broadcast, daemon=True).start()


# ------------------------------------------------------------------ #
# Entry point                                                          #
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    print("=" * 55)
    print("  Elderly Monitoring System  –  Starting…")
    print("  Dashboard → http://127.0.0.1:5000")
    print("=" * 55)
    socketio.run(app, host="0.0.0.0", port=5000, debug=False)
