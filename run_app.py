import sys
import os

# Add the project root to sys.path
sys.path.append(os.getcwd())

try:
    from Main.app_ui import SmartAttendanceApp
except ImportError as e:
    print(f"Error importing app: {e}")
    # Fallback if run directly inside Main (not recommended but defensive)
    sys.path.append(os.path.join(os.getcwd(), 'Main'))
    try:
        from app_ui import SmartAttendanceApp
    except ImportError:
        print("Critical Error: Could not import SmartAttendanceApp.")
        print("Please ensure you are running this script from the root directory 'Smart_Attendance_System'.")
        sys.exit(1)

if __name__ == "__main__":
    app = SmartAttendanceApp()
    app.mainloop()
