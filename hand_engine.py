import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import time
from screeninfo import get_monitors
from collections import deque
import threading
from actions import Actions
import json

class HandEngine(threading.Thread):
    def __init__(self, shared_state):
        super().__init__()
        self.daemon = True
        self.state = shared_state
        
        o = vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path='hand_landmarker.task'),
            num_hands=1,
            running_mode=vision.RunningMode.VIDEO
        )
        self.det = vision.HandLandmarker.create_from_options(o)
        self.scr_w = get_monitors()[0].width
        self.scr_h = get_monitors()[0].height
        
        self.ema_alpha = 0.6
        self.ema_x, self.ema_y = 0, 0
        self.kalman = np.zeros((4, 1))
        self.k_A = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]])
        self.k_H = np.array([[1,0,0,0],[0,1,0,0]])
        self.k_P = np.eye(4)
        
        self.cal_start = 0
        self.hand_sz = 1.0
        
        self.pinch_f = 0
        self.pinch_s = False
        self.last_p = 0
        
        self.prv_z = 0
        self.prv_v = 0
        self.prv_b = 0
        self.prv_sy = 0
        
        self.v_start = 0
        self.v_act = False
        self.v_hist = deque(maxlen=10)

    def reload_config(self):
        try:
            with open('config.json', 'r') as f:
                c = json.load(f)
                self.state['sens'] = float(c.get('sensitivity', 0.05))
                self.ema_alpha = 1.0 - (float(c.get('smoothing', 5)) / 10.0)
                if self.ema_alpha <= 0: self.ema_alpha = 0.1
        except:
            self.state['sens'] = 0.05
            self.ema_alpha = 0.5

    def filter_pos(self, z):
        self.kalman = np.dot(self.k_A, self.kalman)
        self.k_P = np.dot(np.dot(self.k_A, self.k_P), self.k_A.T) + (np.eye(4)*0.1)
        S = np.dot(np.dot(self.k_H, self.k_P), self.k_H.T) + (np.eye(2)*5.0)
        K = np.dot(np.dot(self.k_P, self.k_H.T), np.linalg.inv(S))
        self.kalman += np.dot(K, (z - np.dot(self.k_H, self.kalman)))
        self.k_P -= np.dot(np.dot(K, self.k_H), self.k_P)
        kx, ky = self.kalman[0][0], self.kalman[1][0]
        
        if self.ema_x == 0:
            self.ema_x, self.ema_y = kx, ky
        else:
            self.ema_x = self.ema_alpha * kx + (1 - self.ema_alpha) * self.ema_x
            self.ema_y = self.ema_alpha * ky + (1 - self.ema_alpha) * self.ema_y
            
        return self.ema_x, self.ema_y

    def dist(self, p1, p2):
        return np.linalg.norm(np.array([p1.x-p2.x, p1.y-p2.y]))

    def ang(self, p1, p2):
        return np.degrees(np.arctan2(p2.y-p1.y, p2.x-p1.x))

    def run(self):
        import pythoncom
        pythoncom.CoInitialize()
        self.act = Actions()
        cap = cv2.VideoCapture(0)
        ts = 0
        while cap.isOpened() and self.state.get('run', True):
            self.reload_config()
            self.state['app'] = self.act.get_app_mode()
            s, f = cap.read()
            if not s: continue
            
            ts += 66
            f = cv2.flip(f, 1)
            rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
            mpi = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            res = self.det.detect_for_video(mpi, ts)
            
            act_g = "None"
            
            if res.hand_landmarks:
                lm = res.hand_landmarks[0]
                
                if self.state.get('cal'):
                    if self.cal_start == 0: self.cal_start = time.time()
                    e = time.time() - self.cal_start
                    cv2.putText(f, f"Calib: {5-int(e)}", (50,50), cv2.FONT_HERSHEY_PLAIN, 2, (0,255,0), 2)
                    self.hand_sz = self.dist(lm[0], lm[9])
                    if e > 5:
                        self.state['cal'] = False
                        self.cal_start = 0
                    cv2.imshow("Virtual Mouse", f)
                    cv2.waitKey(1)
                    continue

                ns = self.state.get('sens', 0.05) * (self.hand_sz / 0.3)
                
                i8, m12, t4 = lm[8], lm[12], lm[4]
                r16, p20 = lm[16], lm[20]
                w0 = lm[0]
                
                up = [lm[i].y < lm[i-2].y for i in [8,12,16,20]]
                
                dim = self.dist(i8, m12)
                dti = self.dist(t4, i8)
                dtm = self.dist(t4, m12)
                dtr = self.dist(t4, r16)
                dtp = self.dist(t4, p20)
                
                fx, fy = self.filter_pos(np.array([[i8.x], [i8.y]]))
                gx, gy = w0.x, w0.y 
                
                if dim < ns:
                    self.act.move_cursor(fx*self.scr_w, fy*self.scr_h)
                    act_g = "Move"

                if dti < ns:
                    self.pinch_f += 1
                    if self.pinch_f >= 2 and not self.pinch_s:
                        t = time.time()
                        if t - self.last_p < 0.4:
                            self.act.click(double=True)
                            act_g = "Double Click"
                        else:
                            self.act.click(drag_start=True)
                            act_g = "Drag Start"
                        self.last_p = t
                        self.pinch_s = True
                else:
                    self.pinch_f = 0
                    if self.pinch_s:
                        self.act.click(drag_end=True)
                        act_g = "Drag End"
                    self.pinch_s = False

                if dtm < ns:
                    self.act.click(right=True)
                    act_g = "Right Click"

                if all(up):
                    if self.prv_sy != 0:
                        dy = (gx - self.prv_sy) * 500
                        if self.state.get('app') == "Browser": dy *= 1.5
                        self.act.scroll(int(-dy))
                        act_g = "Scroll"
                    self.prv_sy = gx
                else:
                    self.prv_sy = 0

                zd = self.dist(t4, p20)
                if not up[0] and not up[1] and not up[2] and up[3]:
                    act_g = "Draw Toggle"
                    self.state['draw'] = not self.state.get('draw', False)
                    time.sleep(0.5)

                if dtr < ns:
                    ang = self.ang(w0, lm[9])
                    if self.prv_v != 0:
                        if ang > self.prv_v + 2: self.act.volume(True)
                        elif ang < self.prv_v - 2: self.act.volume(False)
                        act_g = "Volume"
                    self.prv_v = ang
                else:
                    self.prv_v = 0

                if dtp < ns:
                    cy = t4.y
                    if self.prv_b != 0:
                        self.act.brightness(50 + (self.prv_b - cy)*100)
                        act_g = "Brightness"
                    self.prv_b = cy
                else:
                    self.prv_b = 0

                if not any(up):
                    self.act.media('playpause')
                    act_g = "Play/Pause"
                    time.sleep(0.3)

                if up[0] and up[1] and not up[2] and not up[3]:
                    if not self.v_act:
                        self.v_start = time.time()
                        self.v_act = True
                    elif time.time() - self.v_start > 2:
                        self.act.copy()
                        act_g = "Copy"
                        self.v_act = False
                    self.v_hist.append(i8.y)
                    if len(self.v_hist) == 10 and self.v_hist[-1] > self.v_hist[0] + 0.2:
                        self.act.cut()
                        act_g = "Cut"
                        self.v_hist.clear()
                else:
                    self.v_act = False

                if up[0] and up[1] and up[2] and not up[3]:
                    self.act.paste()
                    act_g = "Paste"
                    time.sleep(0.5)

                if self.state.get('draw'):
                    self.state['dx'] = i8.x
                    self.state['dy'] = i8.y

                for landmark in res.hand_landmarks[0]:
                    cv2.circle(f, (int(landmark.x*f.shape[1]), int(landmark.y*f.shape[0])), 3, (0, 255, 0), -1)

            self.state['g'] = act_g
            cv2.imshow("Virtual Mouse", f)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.state['run'] = False
        cap.release()
        cv2.destroyAllWindows()
