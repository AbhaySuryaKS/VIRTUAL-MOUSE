import customtkinter as ctk
import tkinter as tk
import json
from screeninfo import get_monitors

class UIManager:
    def __init__(self, state):
        self.state = state
        self.state['run'] = True
        self.state['cal'] = True
        self.state['draw'] = False
        
        ctk.set_appearance_mode("Dark")
        self.app = ctk.CTk()
        self.app.title("Virtual Mouse Panel")
        self.app.geometry("400x500")
        
        sw = get_monitors()[0].width
        sh = get_monitors()[0].height
        self.draw_root = tk.Toplevel(self.app)
        self.draw_root.overrideredirect(True)
        self.draw_root.attributes("-topmost", True, "-transparentcolor", "white")
        self.draw_root.geometry(f"{sw}x{sh}+0+0")
        self.cvs = tk.Canvas(self.draw_root, bg="white", highlightthickness=0)
        self.cvs.pack(fill="both", expand=True)
        self.sw = sw
        self.sh = sh
        self.pts = []
        
        self.l1 = ctk.CTkLabel(self.app, text="Sensitivity", font=("Arial", 16, "bold"))
        self.l1.pack(pady=10)
        self.s1 = ctk.CTkSlider(self.app, from_=0.01, to=0.15, command=self.save_cfg)
        self.s1.pack()
        
        self.l2 = ctk.CTkLabel(self.app, text="Smoothing", font=("Arial", 16, "bold"))
        self.l2.pack(pady=10)
        self.s2 = ctk.CTkSlider(self.app, from_=1, to=10, command=self.save_cfg)
        self.s2.pack()
        
        self.b1 = ctk.CTkButton(self.app, text="Recalibrate", command=self.rc)
        self.b1.pack(pady=20)
        
        self.lg = ctk.CTkLabel(self.app, text="Gesture: None", font=("Arial", 20))
        self.lg.pack(pady=10)
        
        self.lm = ctk.CTkLabel(self.app, text="Mode: Global", font=("Arial", 14))
        self.lm.pack(pady=10)
        
        self.app.protocol("WM_DELETE_WINDOW", self.close)
        self.load_cfg()
        self.update_loop()

    def load_cfg(self):
        try:
            with open('config.json', 'r') as f:
                c = json.load(f)
                self.s1.set(c.get('sensitivity', 0.05))
                self.s2.set(c.get('smoothing', 5))
        except:
            self.s1.set(0.05)
            self.s2.set(5)
            self.save_cfg()

    def save_cfg(self, _=None):
        with open('config.json', 'w') as f:
            json.dump({'sensitivity': self.s1.get(), 'smoothing': self.s2.get()}, f)

    def rc(self):
        self.state['cal'] = True

    def update_loop(self):
        if not self.state.get('run', False):
            self.close()
            return

        self.lg.configure(text=f"Gesture: {self.state.get('g', 'None')}")
        self.lm.configure(text=f"App: {self.state.get('app', 'Global')}")
        
        if self.state.get('draw'):
            self.draw_root.deiconify()
            if 'dx' in self.state and 'dy' in self.state:
                px = self.state['dx'] * self.sw
                py = self.state['dy'] * self.sh
                self.pts.append((px, py))
                if len(self.pts) > 500: self.pts.pop(0)
                self.cvs.delete("all")
                if len(self.pts) > 1:
                    for i in range(len(self.pts)-1):
                        self.cvs.create_line(self.pts[i], self.pts[i+1], fill="red", width=3)
        else:
            self.draw_root.withdraw()
            self.pts.clear()
            self.cvs.delete("all")
            
        self.app.after(50, self.update_loop)

    def close(self):
        self.state['run'] = False
        self.app.destroy()
        self.draw_root.destroy()
