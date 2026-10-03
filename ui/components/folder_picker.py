import tkinter as tk
from tkinter import filedialog

def open_folder_dialog() -> str:
    """Opens a native OS folder selection window using Tkinter."""
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    folder_selected = filedialog.askdirectory(master=root)
    root.destroy()
    return folder_selected