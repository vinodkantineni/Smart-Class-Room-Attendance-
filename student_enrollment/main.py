"""
Main UI + Enrollment workflow (Tkinter)
"""
import tkinter as tk
import enrollment_utils.config as config
from ui.layout import setup_ui
import traceback

def main():
    print("Initializing Application...")
    try:
        root = tk.Tk()
        root.title(config.APP_TITLE)
        print("UI Setup...")
        setup_ui(root)
        print("Starting Main Loop...")
        root.mainloop()
    except Exception:
        traceback.print_exc()
        input("Press Enter to exit...")

if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
