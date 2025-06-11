import time
import tkinter as tk

class Stopwatch:
    def __init__(self, root):
        self.running = False
        self.start_time = 0
        self.elapsed_time = 0

        self.label = tk.Label(root, text="00:00:00", font=("Helvetica", 40))
        self.label.pack()

        self.start_button = tk.Button(root, text="Start", command=self.start)
        self.start_button.pack()

        self.stop_button = tk.Button(root, text="Stop", command=self.stop)
        self.stop_button.pack()

        self.reset_button = tk.Button(root, text="Reset", command=self.reset)
        self.reset_button.pack()

    def update(self):
        if self.running:
            self.elapsed_time = time.time() - self.start_time
            self.label.config(text=self.format_time(self.elapsed_time))
            root.after(100, self.update)

    def start(self):
        if not self.running:
            self.running = True
            self.start_time = time.time() - self.elapsed_time
            self.update()

    def stop(self):
        if self.running:
            self.running = False

    def reset(self):
        self.running = False
        self.start_time = 0
        self.elapsed_time = 0
        self.label.config(text="00:00:00")

    def format_time(self, seconds):
        minutes, seconds = divmod(int(seconds), 60)
        hours, minutes = divmod(minutes, 60)
        return f"{hours:02}:{minutes:02}:{seconds:02}"

root = tk.Tk()
root.title("Digital Stopwatch")
stopwatch = Stopwatch(root)
root.mainloop()
