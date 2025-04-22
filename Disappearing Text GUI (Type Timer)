import tkinter as tk
from tkinter import messagebox

class WriteOrDieApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Write or Die")

        self.total_time = 12  # total countdown time in seconds
        self.remaining_time = self.total_time

        # Text input area
        self.text = tk.Text(self.root, wrap=tk.WORD, font=("Helvetica", 14))
        self.text.pack(expand=True, fill=tk.BOTH)
        self.text.bind("<Key>", self.reset_timers)

        # Status/warning label
        self.status = tk.Label(self.root, text="Start typing... or else!", anchor="w", fg="white")
        self.status.pack(fill=tk.X)

        # Countdown label
        self.countdown_label = tk.Label(self.root, text=f"Time left: {self.remaining_time}s",
                                        font=("Helvetica", 20, "bold"), fg="red")
        self.countdown_label.pack()

        self.timer_id = None
        self.warning_stages = {
            10: "Keep Typing, Boy!",
            7: "You Better Not Stop",
            3: "FINAL WARNING!"
        }

        self.start_countdown()

    def reset_timers(self, event=None):
        self.remaining_time = self.total_time
        self.update_countdown_label()
        self.status.config(text="You're doing great. Keep going!")

        if self.timer_id:
            self.root.after_cancel(self.timer_id)

        self.start_countdown()

    def start_countdown(self):
        self.update_countdown_label()

        if self.remaining_time in self.warning_stages:
            self.status.config(text=self.warning_stages[self.remaining_time])

        if self.remaining_time <= 0:
            self.time_up()
        else:
            self.remaining_time -= 1
            self.timer_id = self.root.after(1000, self.start_countdown)

    def update_countdown_label(self):
        self.countdown_label.config(text=f"Time left: {self.remaining_time}s")

    def time_up(self):
        self.text.delete("1.0", tk.END)
        self.status.config(text="Too late.")
        self.countdown_label.config(text="Time left: 0s")
        messagebox.showerror("Write or Die", "You stopped typing. Your work is gone.")

if __name__ == "__main__":
    root = tk.Tk()
    app = WriteOrDieApp(root)
    root.mainloop()
