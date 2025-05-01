from dataclasses import dataclass
from typing import Optional, List
from threading import Thread, Event
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog, filedialog
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.interpolate import CubicSpline
from scipy.integrate import cumulative_trapezoid
import csv
from copy import deepcopy
import os
import random
import pathlib
import time
import platform
import vlc
import json


@dataclass
class Keyframe:
    time: float
    speed: float
    interp: str = "linear"
    angle: Optional[float] = None

    def unwrap(self):
        return (self.time, self.speed, self.interp, self.angle)

    def to_dict(self):
        return {
            "time": self.time,
            "speed": self.speed,
            "interp": self.interp,
            "angle": self.angle,
        }

    @classmethod
    def from_dict(cls, d):
        return cls(
            time=d["time"],
            speed=d["speed"],
            interp=d["interp"],
            angle=d.get("angle")
        )


@dataclass
class Track:
    name: str
    keyframes: List[Keyframe]
    theta_times: List[float]
    theta_values: List[float]
    audio_file: str = ""

    def to_dict(self):
        return {
            "name": self.name,
            "audio_file": self.audio_file,
            "keyframes": [kf.to_dict() for kf in self.keyframes],
        }

    @classmethod
    def from_dict(cls, d):
        return cls(
            name=d["name"],
            audio_file=d["audio_file"],
            keyframes=[Keyframe.from_dict(kfd) for kfd in d["keyframes"]],
            theta_times=[],
            theta_values=[],
        )


def parse_number(s):
    if isinstance(s, int) or isinstance(s, float):
        return s
    s = s.strip()
    try:
        f = float(s)
        return f
    except ValueError:
        raise ValueError(f"Cannot parse '{s}' as a number.")


class TtkTimer(Thread):
    """Thread-based timer to call a callback at regular intervals."""

    def __init__(self, callback, tick):
        super().__init__()
        self.callback = callback
        self.tick = tick
        self.stopFlag = Event()

    def run(self):
        while not self.stopFlag.wait(self.tick):
            self.callback()

    def stop(self):
        self.stopFlag.set()


class Player:
    """VLC player embedded within the keyframe editor."""

    def __init__(self, root, angular_editor):
        self.root = root
        self.angular_editor = angular_editor

        # Create player frame
        self.frame = ttk.Frame(self.root)
        self.frame.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=10)

        # Setup VLC
        self.instance = vlc.Instance()
        self.player = self.instance.media_player_new()

        # Video panel (for visual feedback)
        self.videopanel = ttk.Frame(self.frame, height=50)
        self.canvas = tk.Canvas(self.videopanel, bg='black', height=50)
        self.canvas.pack(fill=tk.X, expand=True)
        self.videopanel.pack(fill=tk.X, expand=True)

        # Control panel
        self.ctrlpanel = ttk.Frame(self.frame)

        # Control buttons
        for label, cmd in [("Play", self.on_play), ("Pause", self.on_pause), ("Stop", self.on_stop)]:
            ttk.Button(self.ctrlpanel, text=label,
                       command=cmd).pack(side=tk.LEFT, padx=2)

        # Volume control
        ttk.Label(self.ctrlpanel, text="Volume:").pack(side=tk.LEFT, padx=5)
        self.volume_var = tk.IntVar(value=70)
        self.volslider = tk.Scale(self.ctrlpanel, from_=0, to=100, orient=tk.HORIZONTAL,
                                  variable=self.volume_var, command=self.volume_changed, length=100)
        self.volslider.pack(side=tk.LEFT, padx=5)

        self.ctrlpanel.pack(fill=tk.X)

        # Time slider
        self.time_panel = ttk.Frame(self.frame)
        self.current_time_label = ttk.Label(self.time_panel, text="0:00")
        self.current_time_label.pack(side=tk.LEFT, padx=5)

        self.scale_var = tk.DoubleVar()
        self.timeslider = tk.Scale(self.time_panel, variable=self.scale_var, command=self.time_slider_changed,
                                   from_=0, to=1000, orient=tk.HORIZONTAL, length=500)
        self.timeslider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

        self.total_time_label = ttk.Label(self.time_panel, text="0:00")
        self.total_time_label.pack(side=tk.LEFT, padx=5)

        self.time_panel.pack(fill=tk.X, padx=10, pady=5)

        # Initialize timer for UI updates
        self.timer = TtkTimer(self.update_ui, 0.5)
        self.timer.start()

        # Synchronize timing across tracks
        self.prev_time = 0

        # Don't move slider when user is dragging it
        self.user_is_dragging_slider = False

        self.last_slider_update = time.time()
        self.last_slider_value = ""

    def on_open(self):
        """Open an audio file and update the current track."""
        filepath = filedialog.askopenfilename(
            initialdir=pathlib.Path.cwd(),
            title="Choose audio file",
            filetypes=[("Audio files", "*.mp3 *.wav *.flac *.ogg"),
                       ("All files", "*.*")]
        )

        if filepath:
            # Load the media in VLC
            media = self.instance.media_new(filepath)
            media.parse()
            self.player.set_media(media)

            # Update track name and media info
            current_track = self.angular_editor.tracks[self.angular_editor.current_track_index]
            current_track.audio_file = filepath
            current_track.name = os.path.basename(filepath)
            current_track.keyframes[-1].time = round(
                vlc.libvlc_media_get_duration(media) / 1000, 2)

            # Set up the video output (even for audio files, this helps with media events)
            handle = self.videopanel.winfo_id()
            if platform.system() == 'Windows':
                self.player.set_hwnd(handle)
            else:
                self.player.set_xwindow(handle)

            # Set initial volume
            self.player.audio_set_volume(self.volume_var.get())

            # Update track name to include the audio file if not already done
            if os.path.basename(filepath) not in current_track.name:
                current_track.name = f"{current_track.name} - {os.path.basename(filepath)}"
                # Update the combobox
                self.angular_editor.track_selector['values'] = [
                    t.name for t in self.angular_editor.tracks]
                self.angular_editor.track_selector.current(
                    self.angular_editor.current_track_index)

    def on_play(self):
        """Play the media."""
        media = self.player.get_media()
        if not media:
            self.on_open()
            self.on_play()
        else:
            media.parse()

        if self.player.play() == -1:
            messagebox.showerror("Error", "Unable to play the media.")

        if 0 < self.prev_time < vlc.libvlc_media_get_duration(media):
            self.player.set_time(self.prev_time)

    def on_pause(self):
        """Pause the media."""
        if self.player.get_state() == vlc.State.Playing:
            self.prev_time = self.player.get_time()
        self.player.pause()

    def on_stop(self):
        """Stop the media."""
        self.player.stop()
        self.timeslider.set(0)
        self.current_time_label.config(text="0:00")

    def update_ui(self):
        """Update UI elements based on player state."""
        if self.player.get_state() != vlc.State.Playing:
            return

        if time.time() - self.last_slider_update < 1:
            return

        if self.user_is_dragging_slider:
            return

        # Update position slider
        length = self.player.get_length()
        if length > 0:
            self.timeslider.config(to=length / 1000)

            # Format total time
            total_mins = length // 60000
            total_secs = (length % 60000) // 1000
            self.total_time_label.config(text=f"{total_mins}:{total_secs:02d}")

        # Get current time
        current_time = self.player.get_time()
        if current_time > 0:
            # Format current time
            curr_mins = current_time // 60000
            curr_secs = (current_time % 60000) // 1000
            self.current_time_label.config(text=f"{curr_mins}:{curr_secs:02d}")

            # Update slider if not being dragged
            val = current_time / 1000
            self.last_slider_value = f"{val:.0f}.0"
            self.timeslider.set(val)

    def time_slider_changed(self, event=None):
        """Handle time slider changes."""
        if not self.player.get_media():
            return

        self.user_is_dragging_slider = True

        new_val = self.scale_var.get()
        val_str = str(new_val)

        if val_str != self.last_slider_value:
            self.last_slider_update = time.time()
            self.player.set_time(int(new_val * 1000))

        # Wait briefly and reset dragging flag
        def release_drag_flag():
            time.sleep(0.5)
            self.user_is_dragging_slider = False

        Thread(target=release_drag_flag, daemon=True).start()

    def volume_changed(self, event=None):
        """Handle volume slider changes."""
        volume = self.volume_var.get()
        volume = max(0, min(volume, 100))
        self.player.audio_set_volume(volume)

    def cleanup(self):
        """Stop the timer and release resources."""
        self.timer.stop()
        self.on_stop()


class AngularEditor:
    def __init__(self, root):
        self.root = root
        root.title("Angular Speed Keyframe Editor with Audio Player")
        root.geometry("1200x800")

        self.current_track_index = 0

        self.tracks: List[Track] = []
        self.default_track = Track("",
                                   [
                                       Keyframe(0.0, 0.0, "linear"),
                                       Keyframe(1.0, 60.0, "constant", 0.0),
                                       Keyframe(2.0, -60.0, "cubicspline"),
                                       Keyframe(3.0, -40.0, "cubicspline"),
                                       Keyframe(4.0, 0.0, "end-frame")
                                   ], [], [])
        self.empty_track = Track(
            "", [Keyframe(0.0, 0.0, "linear"), Keyframe(0.1, 0.0, "end-frame")], [], [])
        self.tracks.append(deepcopy(self.default_track))

        self.selected = None
        self.clipboard: List[Keyframe] = None

        self.interp_options = ["linear", "cubicspline", "constant"]

        # Create top frame for track selection and track management buttons
        self.top_frame = ttk.Frame(self.root)
        self.top_frame.pack(fill=tk.X, padx=10, pady=5)

        ttk.Label(self.top_frame, text="Track:").pack(side=tk.LEFT, padx=5)

        self.track_selector = ttk.Combobox(
            self.top_frame, values=[t.name for t in self.tracks], width=40)
        self.track_selector.current(0)
        self.track_selector.bind("<<ComboboxSelected>>", self.change_track)
        self.track_selector.pack(side=tk.LEFT, padx=5)

        # Button frame
        self.button_frame = ttk.Frame(self.top_frame)
        self.button_frame.pack(side=tk.LEFT, padx=10)

        ttk.Button(self.button_frame, text="Add Track",
                   command=self.add_track).pack(side=tk.LEFT, padx=2)
        ttk.Button(self.button_frame, text="Remove Track",
                   command=self.remove_track).pack(side=tk.LEFT, padx=2)
        ttk.Button(self.button_frame, text="Duplicate Track",
                   command=self.duplicate_track).pack(side=tk.LEFT, padx=2)
        ttk.Button(self.button_frame, text="Swap Audio Track",
                   command=self.swap_track).pack(side=tk.LEFT, padx=2)

        # Create middle frame for keyframe editor
        self.middle_frame = ttk.Frame(self.root)
        self.middle_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        self.create_widgets()
        self.update_plot()

        # Copy and Paste
        self.root.bind("<Control-c>", self.copy_keyframes)
        self.root.bind("<Control-v>", self.paste_keyframes)

        # Create context menu
        self.context_menu = tk.Menu(self.root, tearoff=0)
        self.context_menu.add_command(
            label="Delete Keyframe", command=self.delete_selected_keyframe)
        self.context_menu.add_command(
            label="Duplicate Keyframe", command=self.duplicate_selected_keyframe)
        self.context_menu.add_command(
            label="Copy Keyframe", command=self.copy_selected_keyframe)

        self.paste_menu = tk.Menu(self.root, tearoff=0)
        self.paste_menu.add_command(
            label="Paste Keyframe", command=self.paste_selected_keyframe)
        self.tree.bind("<Button-3>", self.on_right_click)

        # Initialize the Player class (will be created at the bottom of the window)
        self.player = Player(self.root, self)
        self.player.on_open()
        self.track_selector['values'] = [t.name for t in self.tracks]
        self.track_selector.current(self.current_track_index)
        self.refresh_tree()
        self.update_plot()

    def add_track(self):
        new_track = deepcopy(self.empty_track)
        self.tracks.append(new_track)
        self.current_track_index = len(self.tracks) - 1
        self.player.on_open()
        if self.tracks[-1].name:
            self.track_selector['values'] = [t.name for t in self.tracks]
            self.track_selector.current(len(self.tracks) - 1)
        else:
            del self.tracks[-1]
        self.change_track(None)
        self.update_plot()

    def remove_track(self):
        if len(self.tracks) <= 1:
            messagebox.showwarning(
                "Cannot Remove", "At least one track must exist.")
            return

        track = self.tracks[self.current_track_index]
        confirm = messagebox.askyesno(
            "Remove Track", f"Are you sure you want to remove '{track.name}'?")
        if confirm:
            del self.tracks[self.current_track_index]
            self.current_track_index = 0
            self.track_selector['values'] = [t.name for t in self.tracks]
            self.track_selector.current(0)
            self.change_track(None)
            self.update_plot()

    def duplicate_track(self):
        self.tracks.append(deepcopy(self.tracks[self.current_track_index]))
        self.track_selector['values'] = [t.name for t in self.tracks]
        self.track_selector.current(len(self.tracks) - 1)
        self.change_track(None)
        self.update_plot()

    def swap_track(self):
        self.player.on_open()
        self.track_selector['values'] = [t.name for t in self.tracks]
        self.track_selector.current(self.current_track_index)
        self.update_plot()

    def change_track(self, event):
        self.current_track_index = self.track_selector.current()
        self.refresh_tree()
        self.player.on_pause()

        # Load associated audio file if it exists
        current_track = self.tracks[self.current_track_index]
        if current_track.audio_file and os.path.exists(current_track.audio_file):
            # prev_time = self.player.player.get_time()
            media = self.player.instance.media_new(current_track.audio_file)
            self.player.player.set_media(media)
            # self.player.player.play()
            # if prev_time < self.player.player.get_length():
            #     self.player.player.set_time(prev_time)
            # self.player.player.pause()
            self.player.volslider.set(self.player.player.audio_get_volume())

    def create_widgets(self):
        # Split into left and right frames
        self.left_frame = ttk.Frame(self.middle_frame)
        self.left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)

        self.right_frame = ttk.Frame(self.middle_frame)
        self.right_frame.pack(side=tk.RIGHT, fill=tk.BOTH,
                              expand=True, padx=5, pady=5)

        # Treeview for keyframes
        self.tree = ttk.Treeview(self.left_frame, columns=(
            "Time", "Speed", "Interp", "Angle"), show='headings', height=15)
        self.tree.heading("Time", text="Time (s)")
        self.tree.heading("Speed", text="Speed (deg/s)")
        self.tree.heading("Interp", text="Interpolation")
        self.tree.heading("Angle", text="Angle (deg)")

        # Set column widths
        self.tree.column("Time", width=80)
        self.tree.column("Speed", width=100)
        self.tree.column("Interp", width=100)
        self.tree.column("Angle", width=80)

        self.tree.bind("<Double-1>", self.on_edit_cell)
        self.tree.bind("<<TreeviewSelect>>", self.on_select)

        # Add scrollbar
        tree_scroll = ttk.Scrollbar(
            self.left_frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=tree_scroll.set)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        tree_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        # Buttons frame for keyframe operations
        self.kf_btn_frame = ttk.Frame(self.left_frame)
        self.kf_btn_frame.pack(fill=tk.X, pady=5)
        self.design_btn_frame = ttk.Frame(self.kf_btn_frame)
        self.design_btn_frame.pack(side=tk.BOTTOM, pady=5)

        # Buttons with better layout
        ttk.Button(self.kf_btn_frame, text="Add Keyframe",
                   command=self.add_keyframe).pack(side=tk.LEFT, padx=5)
        ttk.Button(self.kf_btn_frame, text="Delete Selected",
                   command=self.delete_keyframes).pack(side=tk.LEFT, padx=5)
        ttk.Button(self.kf_btn_frame, text="Export Data",
                   command=self.export_data).pack(side=tk.LEFT, padx=5)
        ttk.Button(self.design_btn_frame, text="Save Design",
                   command=self.save_design).pack(side=tk.LEFT, padx=5)
        ttk.Button(self.design_btn_frame, text="Load Design",
                   command=self.load_design).pack(side=tk.LEFT, padx=5)

        # Matplotlib plot
        self.fig, self.axs = plt.subplots(2, 1, figsize=(8, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.right_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.refresh_tree()

    def refresh_tree(self):
        for i in self.tree.get_children():
            self.tree.delete(i)
        for idx, kf in enumerate(self.tracks[self.current_track_index].keyframes):
            row_id = self.tree.insert('', 'end', values=(
                kf.time, kf.speed, kf.interp, kf.angle))
            if idx == 0 or idx == len(self.tracks[self.current_track_index].keyframes) - 1:
                self.tree.item(row_id, tags=('boundary',))

        self.tree.tag_configure('boundary', background='#e0e0e0')

    def on_select(self, event):
        selected = self.tree.selection()
        if selected:
            self.selected = selected[0]
        else:
            self.selected = None

    def is_valid_keyframe(self, kf) -> bool:
        if not isinstance(kf, Keyframe):
            return False

        if kf.time is None or kf.speed is None:
            return False

        if kf.time < 0.0:
            return False

        if kf.interp not in self.interp_options and kf.interp != "end-frame":
            return False

        return True

    def add_keyframe(self, **kwargs):
        use_track_time = messagebox.askyesno(
            "Add Keyframe", "Use current time in audio track?")
        if use_track_time:
            time = round(self.player.player.get_time() / 1000, 2)
        else:
            time = simpledialog.askfloat(
                "Add Keyframe", "Enter time (sec):")
        speed = simpledialog.askfloat(
            "Add Keyframe", "Enter speed (deg/sec):")
        interp = simpledialog.askstring(
            "Add Keyframe", "Enter interpolation type (default 'linear'):")
        angle = simpledialog.askstring(
            "Add Keyframe", "Enter new angle (deg) (optional):")
        if not angle:
            angle = None
        else:
            try:
                angle = parse_number(angle)
            except Exception as ex:
                messagebox.showerror("Invalid Angle", str(ex))

        if interp == "cubicspline":
            messagebox.showwarning(
                "Warning", "For sequential cubicspline keyframes, only first angle is applied."
            )
            angle = None

        if interp not in self.interp_options:
            interp = self.interp_options[0]

        if self.tracks[self.current_track_index].keyframes[-1].time < time:
            messagebox.showerror(
                "Invalid Time", "Cannot have keyframe later than end of audio.")
            return

        if any(kf.time == time for kf in self.tracks[self.current_track_index].keyframes):
            messagebox.showerror(
                "Invalid Time", "Cannot have two keyframes at same time.")
            return

        new_kf = Keyframe(time, speed, interp, angle)

        if self.is_valid_keyframe(new_kf):
            self.add_to_current_track(new_kf)
            self.sort_current_track()
        self.ensure_boundaries()
        self.refresh_tree()
        self.update_plot()

    def delete_keyframes(self):
        selected = self.tree.selection()
        if not selected:
            return

        if len(self.tracks[self.current_track_index].keyframes) - len(selected) < 2:
            messagebox.showwarning("Cannot Delete", f"Cannot delete {len(selected)} keyframes. Each track must have 2 keyframes.")
            return

        for row_id in selected:
            values = self.tree.item(row_id)['values']
            time = float(values[0])

            self.tracks[self.current_track_index].keyframes = [
                kf for kf in self.tracks[self.current_track_index].keyframes if not kf.time == time]

        self.sort_current_track()
        self.refresh_tree()
        self.update_plot()

    def copy_keyframes(self, event):
        selected = self.tree.selection()
        if not selected:
            return
        else:
            self.clipboard = []

        for row_id in selected:
            index = self.tree.index(row_id)
            kf = self.tracks[self.current_track_index].keyframes[index]
            self.clipboard.append(kf)

    def paste_keyframes(self, event):
        for new_kf in self.clipboard:
            if any(kf.time == new_kf.time for kf in self.tracks[self.current_track_index].keyframes):
                messagebox.showerror(
                    "Invalid Time", "Cannot have two keyframes at same time.")
            else:
                self.add_to_current_track(new_kf)
                self.sort_current_track()
                self.ensure_boundaries()
                self.refresh_tree()
                self.update_plot()

    def on_edit_cell(self, event):
        region = self.tree.identify("region", event.x, event.y)
        if region != "cell":
            return

        row_id = self.tree.identify_row(event.y)
        col_id = self.tree.identify_column(event.x)

        if not row_id or not col_id:
            return

        col_index = int(col_id[1:]) - 1
        row_index = self.tree.index(row_id)
        bbox = self.tree.bbox(row_id, col_id)
        if not bbox:
            return

        x, y, width, height = self.tree.bbox(row_id, col_id)

        # Handle interpolation column (dropdown)
        # Interpolation column
        if col_index == 2 and row_index != len(self.tracks[self.current_track_index].keyframes) - 1:
            combo = ttk.Combobox(
                self.tree, values=self.interp_options, state="readonly")
            combo.place(x=x, y=y, width=width, height=height)

            current_value = self.tree.set(row_id, col_id)
            combo.set(
                current_value if current_value in self.interp_options else self.interp_options[0])

            def finish_edit(apply=True):
                if apply:
                    new_value = combo.get()
                    self.tree.set(row_id, col_id, new_value)

                    # Update the keyframe data
                    kf = self.tracks[self.current_track_index].keyframes[row_index]
                    kf.interp = new_value
                    if self.is_valid_keyframe(kf):
                        self.tracks[self.current_track_index].keyframes[row_index] = kf
                    self.ensure_boundaries()
                    self.update_plot()
                combo.destroy()

            def on_select(event=None):
                finish_edit()

            def on_key(event):
                if event.keysym == "Return":
                    finish_edit()
                elif event.keysym == "Escape":
                    finish_edit(apply=False)

            combo.bind("<<ComboboxSelected>>", on_select)
            combo.bind("<Key>", on_key)
            combo.focus_set()

        # Handle numeric columns (time, omega)
        elif col_index != 3 or row_index != len(self.tracks[self.current_track_index].keyframes) - 1:
            current_values = list(self.tree.item(row_id, "values"))

            entry = ttk.Entry(self.tree)
            entry.place(x=x, y=y, width=width, height=height)
            entry.insert(0, current_values[col_index])
            entry.focus()

            def on_enter(e=None):
                val = entry.get()
                current_values[col_index] = val
                try:
                    angle = current_values[3]
                    if angle == 'None':
                        angle = None
                    else:
                        angle = parse_number(angle)
                    parsed = Keyframe(
                        parse_number(current_values[0]),
                        parse_number(current_values[1]),
                        current_values[2],
                        angle
                    )
                    if self.is_valid_keyframe(parsed):
                        self.tracks[self.current_track_index].keyframes[row_index] = parsed
                        self.sort_current_track()
                    self.ensure_boundaries()
                    self.refresh_tree()
                    self.update_plot()
                except Exception as ex:
                    messagebox.showerror("Edit Error", str(ex))
                finally:
                    entry.destroy()

            entry.bind("<Return>", on_enter)
            entry.bind("<FocusOut>", lambda e: entry.destroy())

    def on_right_click(self, event):
        row_id = self.tree.identify_row(event.y)
        if row_id:
            self.tree.selection_set(row_id)
            self.context_menu.post(event.x_root, event.y_root)
        else:
            self.paste_menu.post(event.x_root, event.y_root)

        # Bind left-click to dismiss context menu when clicking anywhere else
        self.root.bind("<Button-1>", self.dismiss_context_menu)

    def dismiss_context_menu(self, event):
        self.context_menu.unpost()  # This closes the context menu
        self.root.unbind("<Button-1>")  # Unbind the left-click

    def delete_selected_keyframe(self):
        selected = self.tree.selection()
        if not selected:
            return
        index = self.tree.index(selected[0])

        if len(self.tracks[self.current_track_index].keyframes) < 3:
            messagebox.showwarning(
                "Cannot Delete", "Cannot delete boundary keyframes.")
            return

        del self.tracks[self.current_track_index].keyframes[index]
        self.ensure_boundaries()
        self.refresh_tree()
        self.update_plot()

    def duplicate_selected_keyframe(self):
        selected = self.tree.selection()
        if not selected:
            return
        index = self.tree.index(selected[0])
        print(selected)
        kf = self.tracks[self.current_track_index].keyframes[index]
        new_t = kf.time + 0.1

        self.add_to_current_track(
            Keyframe(new_t, kf.speed, kf.interp, kf.angle))
        self.sort_current_track()
        self.ensure_boundaries()
        self.refresh_tree()
        self.update_plot()

    def copy_selected_keyframe(self):
        selected = self.tree.selection()
        if not selected:
            return
        index = self.tree.index(selected[0])
        self.clipboard = [
            self.tracks[self.current_track_index].keyframes[index]]

    def paste_selected_keyframe(self):
        if any(kf.time == self.clipboard[0].time for kf in self.tracks[self.current_track_index].keyframes):
            messagebox.showerror(
                "Invalid Time", "Cannot have two keyframes at same time.")
            return
        else:
            self.add_to_current_track(self.clipboard[0])
            self.sort_current_track()
            self.ensure_boundaries()
            self.refresh_tree()
            self.update_plot()

    def add_to_current_track(self, kf):
        self.tracks[self.current_track_index].keyframes.append(kf)

    def sort_current_track(self):
        self.tracks[self.current_track_index].keyframes.sort(
            key=lambda kf: kf.time)

    def ensure_boundaries(self):
        for idx, kf in enumerate(self.tracks[self.current_track_index].keyframes):
            if idx != len(self.tracks[self.current_track_index].keyframes) - 1 and kf.interp == "end-frame":
                kf.interp = "linear"
        self.tracks[self.current_track_index].keyframes[-1].interp = "end-frame"
        self.tracks[self.current_track_index].keyframes[-1].angle = None

    def update_plot(self):
        colors = ["tab:blue", "tab:orange",
                  "tab:green", "tab:red", "tab:purple"]

        self.fig.clf()
        ax_speed = self.fig.add_subplot(2, 1, 1)
        ax_theta = self.fig.add_subplot(2, 1, 2)

        for idx, track in enumerate(self.tracks):
            if len(track.keyframes) < 2:
                continue

            keyframes = sorted(track.keyframes, key=lambda kf: kf.time)

            omega_times = []
            omega_values = []
            theta_times = []
            theta_values = []

            theta_cursor = 0.0
            i = 0
            while i < len(keyframes) - 1:
                t0, w0, interp, angle = keyframes[i].unwrap()
                if angle is not None:
                    theta_cursor = angle
                t1, w1, _, _ = keyframes[i + 1].unwrap()

                N = round((t1 - t0) * 20)
                if N <= 1:
                    i += 1
                    continue

                t_segment = np.linspace(t0, t1, N, endpoint=False)

                if interp == "linear":
                    omega_segment = np.linspace(w0, w1, N, endpoint=False)

                elif interp == "constant":
                    omega_segment = np.full_like(t_segment, w0)

                elif interp == "cubicspline":
                    j = i
                    while j + 1 < len(keyframes) and keyframes[j].interp == "cubicspline":
                        j += 1
                    spline_times = [keyframes[k].time for k in range(i, j + 1)]
                    spline_speeds = [
                        keyframes[k].speed for k in range(i, j + 1)]
                    spline = CubicSpline(spline_times, spline_speeds)
                    t_segment = np.linspace(spline_times[0], spline_times[-1], round(
                        (spline_times[-1] - spline_times[0]) * 20), endpoint=False)
                    omega_segment = spline(t_segment)
                    i = j  # Skip to end of spline section
                elif interp == "end-frame":
                    break
                else:
                    messagebox.showerror(
                        "Interpolation Error", f"Unknown interpolation: {interp}")
                    i += 1
                    continue

                theta_segment = theta_cursor + \
                    cumulative_trapezoid(omega_segment, t_segment, initial=0)
                theta_cursor = theta_segment[-1]

                omega_times.extend(t_segment)
                omega_values.extend(omega_segment)
                theta_times.extend(t_segment)
                theta_values.extend(theta_segment)

                if interp != "cubicspline":
                    i += 1

            ax_speed.plot(omega_times, omega_values,
                          label=f"ω - {track.name.split('/')[-1]}", color=colors[idx % len(colors)])
            ax_theta.plot(theta_times, theta_values,
                          label=f"θ - {track.name.split('/')[-1]}", color=colors[idx % len(colors)])

            self.tracks[idx].theta_times = theta_times
            self.tracks[idx].theta_values = theta_values

        ax_speed.set_title("Angular Speed ω(t)")
        ax_speed.set_xlabel("Time (s)")
        ax_speed.set_ylabel("ω (rad/s)")
        ax_speed.grid(True)
        ax_speed.legend()

        ax_theta.set_title("Angular Position θ(t)")
        ax_theta.set_xlabel("Time (s)")
        ax_theta.set_ylabel("θ (rad)")
        ax_theta.grid(True)
        ax_theta.legend()

        self.canvas.draw()

    def export_data(self, filename=None):
        filename = simpledialog.askstring("Export Data", "Enter filename:")

        # Step 4: Export results (print or save to a file)
        if filename is not None:
            with open(filename, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["SOUNDSCAPE_LOCATION_FILE_v1"])
                writer.writerow([filename])
                writer.writerow(["50ms"])
                for t in self.tracks:
                    writer.writerow([t.name])
                    writer.writerow(
                        [str(round(t.theta_times[0] * 1000)) + "ms"])
                    writer.writerow(["#" + f'{random.randint(0, 255):02x}' +
                                     f'{random.randint(0, 255):02x}' + f'{random.randint(0, 255):02x}'])
                    for i in range(len(t.theta_times)):
                        writer.writerow([round((t.theta_times[i] - t.theta_times[0]) * 1000),
                                         t.theta_values[i] % 360])
                    writer.writerow([])
            print(f"Data exported to {filename}")
        else:
            print("50ms")
            for t in self.tracks:
                print(t.name)
                print(f"{round(t.theta_times[0] * 1000)}ms")
                print("#" + f'{random.randint(0, 255):02x}' +
                      f'{random.randint(0, 255):02x}' + f'{random.randint(0, 255):02x}')
                for i in range(len(t.theta_times)):
                    print(f"{round((t.theta_times[i] - t.theta_times[0]) * 1000)},{round(t.theta_values[i])}")
                print()

    def save_design(self):
        filename = filedialog.asksaveasfilename(
            initialdir=pathlib.Path.cwd(),
            title="Choose save location",
            filetypes=[("Soundscape design file", "*.sscp_des"),
                       ("All files", "*.*")]
        )

        if filename is None:
            return

        with open(filename, 'w') as f:
            json.dump([t.to_dict() for t in self.tracks], f, indent=2)

    def load_design(self):
        filename = filedialog.askopenfilename(
            initialdir=pathlib.Path.cwd(),
            title="Choose design file to load",
            filetypes=[("Soundscape design file", "*.sscp_des"),
                       ("All files", "*.*")]
        )

        if filename is None:
            return

        with open(filename, 'r') as f:
            data = json.load(f)
            self.tracks = [Track.from_dict(td) for td in data]

        self.track_selector['values'] = [t.name for t in self.tracks]
        self.track_selector.current(0)
        self.change_track(None)
        self.update_plot()


# Run it
if __name__ == "__main__":
    root = tk.Tk()
    app = AngularEditor(root)
    root.mainloop()
