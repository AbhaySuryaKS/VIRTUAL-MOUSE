import pyautogui
import win32api
import win32con
import win32gui
import pygetwindow as gw
import win32com.client

class Actions:
    def __init__(self):
        pyautogui.FAILSAFE = False
        pyautogui.PAUSE = 0
        try:
            self.wmi = win32com.client.GetObject("winmgmts:\\\\.\\root\\WMI").InstancesOf("WmiMonitorBrightnessMethods")
        except:
            self.wmi = None

    def get_app_mode(self):
        try:
            title = win32gui.GetWindowText(win32gui.GetForegroundWindow())
            if any(n in title.lower() for n in ["chrome", "edge", "firefox", "browser"]):
                return "Browser"
            if any(n in title.lower() for n in ["player", "vlc", "spotify", "media"]):
                return "Media Player"
            return "Global"
        except:
            return "Global"

    def move_cursor(self, x, y):
        pyautogui.moveTo(x, y)

    def click(self, right=False, double=False, drag_start=False, drag_end=False):
        if double: pyautogui.doubleClick()
        elif right: pyautogui.rightClick()
        elif drag_start: pyautogui.mouseDown()
        elif drag_end: pyautogui.mouseUp()
        else: pyautogui.click()

    def scroll(self, amount):
        pyautogui.scroll(amount)

    def zoom(self, in_zoom=True):
        pyautogui.hotkey('ctrl', '+' if in_zoom else '-')

    def volume(self, up=True):
        cmd = win32con.VK_VOLUME_UP if up else win32con.VK_VOLUME_DOWN
        try:
            win32api.keybd_event(cmd, 0)
            win32api.keybd_event(cmd, 0, win32con.KEYEVENTF_KEYUP)
        except:
            pass

    def brightness(self, level):
        if self.wmi:
            level = max(0, min(100, int(level)))
            for monitor in self.wmi:
                try:
                    monitor.WmiSetBrightness(0, level)
                except:
                    pass

    def copy(self): pyautogui.hotkey('ctrl', 'c')
    def paste(self): pyautogui.hotkey('ctrl', 'v')
    def cut(self): pyautogui.hotkey('ctrl', 'x')

    def media(self, action):
        keys = {'playpause': 'playpause', 'next': 'nexttrack', 'prev': 'prevtrack'}
        if action in keys:
            pyautogui.press(keys[action])
