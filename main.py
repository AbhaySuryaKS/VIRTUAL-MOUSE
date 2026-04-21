from hand_engine import HandEngine
from ui_manager import UIManager

if __name__ == "__main__":
    s = {}
    h = HandEngine(s)
    h.start()
    u = UIManager(s)
    try:
        u.app.mainloop()
    except KeyboardInterrupt:
        s['run'] = False
        try: u.app.destroy()
        except: pass