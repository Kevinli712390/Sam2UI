import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
import os
import threading
import json
import sys
import tempfile
import shutil
from pathlib import Path
import traceback
from omegaconf import OmegaConf, DictConfig

# Add SAM2 path to Python path
SAM2_PATH = r"C:\Users\kevin\segment-anything-2"
if SAM2_PATH not in sys.path:
    sys.path.append(SAM2_PATH)

# Import torch for device detection
try:
    import torch
except ImportError:
    torch = None

class SAM2VideoUI:
    def __init__(self, root):
        self.root = root
        self.root.title("SAM2 Video Segmentation Tool")
        self.root.geometry("1400x900")
        self.root.configure(bg='#2b2b2b')
        
        # Hardcoded paths
        self.sam2_base_path = r"C:\Users\kevin\segment-anything-2"
        self.checkpoint_dir = r"C:\Users\kevin\segment-anything-2\checkpoints"
        self.config_dir = r"C:\Users\kevin\segment-anything-2\configs"
        
        # Variables
        self.video_path = None
        self.video_cap = None
        self.frames = []
        self.current_frame_idx = 0
        self.current_frame = None
        self.display_frame = None
        self.scale_factor = 1.0
        self.click_points = []  # Store click coordinates with object IDs
        self.masks = {}  # Store masks for each frame {frame_idx: {obj_id: mask}}
        self.playing = False
        self.inference_state = None
        self.current_object_id = 1  # Currently selected object ID
        self.max_object_id = 1  # Track highest object ID used
        self.object_colors = {  # Colors for different objects
            1: [0, 255, 255],    # Cyan
            2: [255, 0, 0],      # Red  
            3: [0, 255, 0],      # Green
            4: [0, 0, 255],      # Blue
            5: [255, 255, 0],    # Yellow
            6: [255, 0, 255],    # Magenta
            7: [255, 165, 0],    # Orange
            8: [128, 0, 128],    # Purple
        }
        
        # SAM2 model
        self.sam2_model = None
        self.model_loaded = False
        
        # UI styling
        self.setup_styles()
        self.setup_ui()
        
    def setup_styles(self):
        """Configure ttk styles"""
        style = ttk.Style()
        style.theme_use('clam')
        
        # Configure colors
        style.configure('TFrame', background='#2b2b2b')
        style.configure('TLabel', background='#2b2b2b', foreground='white')
        style.configure('TButton', background='#404040', foreground='white')
        style.map('TButton', 
                 background=[('active', '#505050'), ('pressed', '#303030')])
        
    def setup_ui(self):
        # Main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)
        
        # Title
        title_label = ttk.Label(main_frame, text="SAM2 Video Segmentation Tool", 
                               font=('Arial', 16, 'bold'))
        title_label.pack(pady=(0, 15))
        
        # Control panel
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill=tk.X, pady=(0, 15))
        
        # File operations
        file_frame = ttk.Frame(control_frame)
        file_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(file_frame, text="📁 Load Video", 
                  command=self.load_video, width=15).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(file_frame, text="🤖 Load SAM2 Model", 
                  command=self.load_sam2_model, width=15).pack(side=tk.LEFT, padx=(0, 10))
        
        # Model status indicator
        self.model_status_label = ttk.Label(file_frame, text="❌ Model Not Loaded", 
                                           foreground='red')
        self.model_status_label.pack(side=tk.LEFT, padx=(20, 0))
        
        # Segmentation controls
        seg_frame = ttk.Frame(control_frame)
        seg_frame.pack(fill=tk.X)
        
        # Object selection
        obj_frame = ttk.Frame(seg_frame)
        obj_frame.pack(fill=tk.X, pady=(0, 5))
        
        ttk.Label(obj_frame, text="Current Object:").pack(side=tk.LEFT)
        
        self.object_var = tk.IntVar(value=1)
        self.object_spinbox = tk.Spinbox(obj_frame, from_=1, to=8, textvariable=self.object_var,
                                        width=5, command=self.on_object_change,
                                        bg='#404040', fg='white', insertbackground='white')
        self.object_spinbox.pack(side=tk.LEFT, padx=(5, 10))
        
        ttk.Button(obj_frame, text="➕ New Object", 
                  command=self.add_new_object, width=12).pack(side=tk.LEFT, padx=(0, 10))
        
        # Object color indicator
        self.object_color_label = ttk.Label(obj_frame, text="●", foreground='cyan', font=('Arial', 16))
        self.object_color_label.pack(side=tk.LEFT, padx=(5, 0))
        
        # Segmentation buttons
        seg_buttons_frame = ttk.Frame(seg_frame)
        seg_buttons_frame.pack(fill=tk.X, pady=(5, 0))
        
        ttk.Button(seg_buttons_frame, text="🎯 Segment Video", 
                  command=self.segment_video, width=15).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(seg_buttons_frame, text="🗑️ Clear Points", 
                  command=self.clear_points, width=15).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(seg_buttons_frame, text="🗑️ Clear Object", 
                  command=self.clear_current_object, width=15).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(seg_buttons_frame, text="💾 Export Masks", 
                  command=self.export_masks, width=15).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(seg_buttons_frame, text="🎬 Export Video", 
                  command=self.export_video, width=15).pack(side=tk.LEFT, padx=(0, 10))
        
        # Show masks toggle
        self.show_masks_var = tk.BooleanVar()
        masks_frame = ttk.Frame(seg_frame)
        masks_frame.pack(fill=tk.X, pady=(5, 0))
        
        ttk.Checkbutton(masks_frame, text="Show Masks Overlay", 
                       variable=self.show_masks_var,
                       command=self.toggle_mask_display).pack(side=tk.LEFT)
        
        # Individual object toggles
        self.object_visibility = {}
        for i in range(1, 9):
            self.object_visibility[i] = tk.BooleanVar(value=True)
            cb = ttk.Checkbutton(masks_frame, text=f"Obj{i}", 
                               variable=self.object_visibility[i],
                               command=self.toggle_mask_display)
            if i <= 4:  # First row
                cb.pack(side=tk.LEFT, padx=(20 if i == 1 else 5, 0))
        
        # Video display area
        display_frame = ttk.Frame(main_frame)
        display_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 15))
        
        # Canvas with scrollbars
        canvas_container = ttk.Frame(display_frame)
        canvas_container.pack(fill=tk.BOTH, expand=True)
        
        # Canvas for video display
        self.canvas = tk.Canvas(canvas_container, bg='#1a1a1a', highlightthickness=0)
        
        # Scrollbars
        v_scrollbar = ttk.Scrollbar(canvas_container, orient=tk.VERTICAL, command=self.canvas.yview)
        h_scrollbar = ttk.Scrollbar(canvas_container, orient=tk.HORIZONTAL, command=self.canvas.xview)
        
        self.canvas.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)
        
        # Pack scrollbars and canvas
        v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Bind canvas events
        self.canvas.bind("<Button-1>", self.on_canvas_click)
        self.canvas.bind("<Button-3>", self.on_canvas_right_click)  # Right click for negative points
        
        # Video controls
        controls_frame = ttk.Frame(main_frame)
        controls_frame.pack(fill=tk.X, pady=(0, 15))
        
        # Playback controls
        playback_frame = ttk.Frame(controls_frame)
        playback_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.play_button = ttk.Button(playback_frame, text="▶️ Play", command=self.toggle_play)
        self.play_button.pack(side=tk.LEFT, padx=(0, 5))
        
        ttk.Button(playback_frame, text="⏮️ Prev", command=self.prev_frame).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(playback_frame, text="⏭️ Next", command=self.next_frame).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(playback_frame, text="⏹️ Reset", command=self.reset_video).pack(side=tk.LEFT, padx=(10, 0))
        
        # Frame slider
        slider_frame = ttk.Frame(controls_frame)
        slider_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(slider_frame, text="Frame:").pack(side=tk.LEFT)
        
        self.frame_var = tk.IntVar()
        self.frame_slider = ttk.Scale(slider_frame, from_=0, to=100, 
                                     orient=tk.HORIZONTAL, variable=self.frame_var,
                                     command=self.on_slider_change)
        self.frame_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(10, 10))
        
        self.frame_label = ttk.Label(slider_frame, text="0/0")
        self.frame_label.pack(side=tk.RIGHT)
        
        # Info panel
        info_frame = ttk.Frame(main_frame)
        info_frame.pack(fill=tk.X)
        
        # Click points info
        points_frame = ttk.Frame(info_frame)
        points_frame.pack(fill=tk.X, pady=(0, 5))
        
        ttk.Label(points_frame, text="Click Points:", font=('Arial', 9, 'bold')).pack(side=tk.LEFT)
        self.points_label = ttk.Label(points_frame, text="None (Left click: +, Right click: -)")
        self.points_label.pack(side=tk.LEFT, padx=(10, 0))
        
        # Status bar
        status_frame = ttk.Frame(info_frame)
        status_frame.pack(fill=tk.X)
        
        ttk.Label(status_frame, text="Status:", font=('Arial', 9, 'bold')).pack(side=tk.LEFT)
        self.status_label = ttk.Label(status_frame, text="Ready - Load a video and SAM2 model to begin")
        self.status_label.pack(side=tk.LEFT, padx=(10, 0))
        
        # Progress bar (initially hidden)
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(info_frame, variable=self.progress_var, maximum=100)
        
    def load_video(self):
        """Load video file and extract frames"""
        file_path = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[
                ("Video files", "*.mp4 *.avi *.mov *.mkv *.flv *.wmv *.m4v"),
                ("MP4 files", "*.mp4"),
                ("AVI files", "*.avi"),
                ("All files", "*.*")
            ]
        )
        
        if not file_path:
            return
            
        try:
            self.video_path = file_path
            self.load_video_frames()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load video: {str(e)}")
            
    def load_video_frames(self):
        """Extract all frames from video"""
        try:
            self.video_cap = cv2.VideoCapture(self.video_path)
            if not self.video_cap.isOpened():
                raise ValueError("Could not open video file")
            
            self.frames = []
            self.masks = {}
            self.click_points = []
            
            # Get video properties
            total_frames = int(self.video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = self.video_cap.get(cv2.CAP_PROP_FPS)
            
            self.status_label.config(text=f"Loading {total_frames} frames...")
            self.progress_bar.pack(fill=tk.X, pady=(5, 0))
            self.root.update()
            
            frame_count = 0
            while True:
                ret, frame = self.video_cap.read()
                if not ret:
                    break
                    
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.frames.append(frame_rgb)
                frame_count += 1
                
                # Update progress
                progress = (frame_count / total_frames) * 100
                self.progress_var.set(progress)
                self.root.update_idletasks()
            
            self.video_cap.release()
            self.progress_bar.pack_forget()
            
            if self.frames:
                self.current_frame_idx = 0
                self.frame_slider.config(to=len(self.frames)-1)
                self.display_current_frame()
                self.status_label.config(text=f"Video loaded: {len(self.frames)} frames @ {fps:.1f} FPS")
                self.update_points_display()
            else:
                raise ValueError("No frames could be extracted from video")
                
        except Exception as e:
            self.progress_bar.pack_forget()
            raise e
            
    def _rgb_to_hex(self, rgb):
        """Convert RGB color to hex"""
        return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"
    
    def on_object_change(self):
        """Handle object selection change"""
        self.current_object_id = self.object_var.get()
        self.update_object_color_display()
        if self.frames:
            self.display_current_frame()
    
    def add_new_object(self):
        """Add a new object for segmentation"""
        if self.max_object_id < 8:
            self.max_object_id += 1
            self.current_object_id = self.max_object_id
            self.object_var.set(self.current_object_id)
            self.object_spinbox.config(to=self.max_object_id)
            self.update_object_color_display()
            self.status_label.config(text=f"Added new object {self.current_object_id}. Click to add points.")
        else:
            messagebox.showwarning("Limit Reached", "Maximum 8 objects supported.")
    
    def update_object_color_display(self):
        """Update the object color indicator"""
        color = self.object_colors[self.current_object_id]
        color_hex = self._rgb_to_hex(color)
        self.object_color_label.config(foreground=color_hex)

    def load_sam2_model(self):
        """Load SAM2 model with explicit top-level config"""
        try:
            self.status_label.config(text="Loading SAM2 model...")
            self.model_status_label.config(text="⏳ Loading...", foreground='orange')
            self.root.update()

            # Explicit paths (ignore configs/sam2/)
            sam2_checkpoint = os.path.join(self.checkpoint_dir, "sam2_hiera_small.pt")
            model_cfg = os.path.join(self.sam2_base_path, "configs", "sam2_hiera_s.yaml")

            # Check files
            if not os.path.exists(sam2_checkpoint):
                raise FileNotFoundError(f"Checkpoint not found: {sam2_checkpoint}")
            if not os.path.exists(model_cfg):
                raise FileNotFoundError(f"Config not found: {model_cfg}")

            # Import builder
            from sam2.build_sam import build_sam2_video_predictor
            import torch

            # Select device
            if torch.cuda.is_available():
                device = "cuda"
                self.status_label.config(text="Using CUDA GPU for inference...")
            else:
                device = "cpu"
                self.status_label.config(text="Using CPU for inference (slower)...")

            print(f"Loading model: {sam2_checkpoint}")
            print(f"Using config: {model_cfg}")
            print(f"Device: {device}")

            from omegaconf import OmegaConf

            # Load YAML manually as dict
            cfg = OmegaConf.load(model_cfg)

            self.sam2_model = build_sam2_video_predictor(
                cfg,
                sam2_checkpoint,
               device=device
            )

            self.model_loaded = True

            model_name = os.path.basename(sam2_checkpoint).replace('.pt', '')
            self.model_status_label.config(text=f"✅ {model_name} ({device.upper()})", foreground='green')
            self.status_label.config(text=f"SAM2 model loaded successfully on {device.upper()}")
        
        except Exception as e:
            self.model_status_label.config(text="❌ Load Failed", foreground='red')

            # Print full traceback to console
            print("=== SAM2 MODEL LOAD FAILED ===")
            traceback.print_exc()
            print("=== END TRACEBACK ===")

            error_msg = f"Failed to load SAM2 model.\n\nFull error: {str(e)}\n\n"
            error_msg += "Make sure:\n"
            error_msg += "1. SAM2 is properly installed\n"
            error_msg += "2. Model checkpoint is in checkpoints/\n"
            error_msg += "3. You are using the TOP-LEVEL config (not configs/sam2/*)\n"
            error_msg += "4. PyTorch is installed with CUDA if using GPU"
            messagebox.showerror("Model Loading Error", error_msg)


    def display_current_frame(self):
        """Display current video frame with overlays - Updated for multiple objects"""
        if not self.frames:
            return
            
        self.current_frame = self.frames[self.current_frame_idx].copy()
        display_frame = self.current_frame.copy()
        
        # Apply mask overlay if enabled and masks exist
        if self.show_masks_var.get() and self.current_frame_idx in self.masks:
            frame_masks = self.masks[self.current_frame_idx]
            
            # Apply each object's mask with its color
            for obj_id, mask in frame_masks.items():
                if obj_id in self.object_visibility and self.object_visibility[obj_id].get():
                    if len(mask.shape) == 2:  # Single channel mask
                        # Get object color
                        obj_color = self.object_colors.get(obj_id, [255, 255, 255])
                        
                        # Create colored overlay
                        overlay = np.zeros_like(display_frame)
                        overlay[mask > 0] = obj_color
                        
                        # Blend with current frame
                        alpha = 0.4
                        display_frame = cv2.addWeighted(display_frame, 1-alpha, overlay, alpha, 0)
        
        # Draw click points
        for i, (x, y, is_positive, obj_id) in enumerate(self.click_points):
            # Only draw points for current object or if showing all
            if obj_id == self.current_object_id or not hasattr(self, 'current_object_id'):
                obj_color = self.object_colors.get(obj_id, [255, 255, 255])
                color = tuple(obj_color) if is_positive else (255, 0, 0)  # Object color for positive, red for negative
                symbol = "+" if is_positive else "-"
                
                # Draw circle
                cv2.circle(display_frame, (int(x), int(y)), 8, color, -1)
                cv2.circle(display_frame, (int(x), int(y)), 10, (255, 255, 255), 2)
                
                # Draw symbol
                cv2.putText(display_frame, symbol, (int(x)-5, int(y)+5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Draw object ID
                cv2.putText(display_frame, f"O{obj_id}", (int(x)+15, int(y)-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 2)
        
        # Convert to PIL and display
        pil_image = Image.fromarray(display_frame)
        
        # Scale image to fit canvas
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        if canvas_width > 1 and canvas_height > 1:
            img_width, img_height = pil_image.size
            
            # Calculate scale to fit canvas while maintaining aspect ratio
            scale_w = (canvas_width - 20) / img_width
            scale_h = (canvas_height - 20) / img_height
            self.scale_factor = min(scale_w, scale_h, 1.0)
            
            new_width = int(img_width * self.scale_factor)
            new_height = int(img_height * self.scale_factor)
            
            pil_image = pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Convert to PhotoImage and display
        self.display_frame = ImageTk.PhotoImage(pil_image)
        self.canvas.delete("all")
        
        # Center the image on canvas
        canvas_center_x = canvas_width // 2
        canvas_center_y = canvas_height // 2
        self.canvas.create_image(canvas_center_x, canvas_center_y, image=self.display_frame)
        
        # Update scroll region
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        
        # Update frame info
        self.frame_var.set(self.current_frame_idx)
        self.frame_label.config(text=f"{self.current_frame_idx + 1}/{len(self.frames)}")
        
    def on_canvas_click(self, event):
        """Handle left mouse click (positive point)"""
        self.add_click_point(event, is_positive=True)
        
    def on_canvas_right_click(self, event):
        """Handle right mouse click (negative point)"""
        self.add_click_point(event, is_positive=False)
        
    def add_click_point(self, event, is_positive=True):
        """Add click point for segmentation"""
        if not self.frames or not self.current_frame_idx < len(self.frames):
            return
            
        # Get canvas coordinates
        canvas_x = self.canvas.canvasx(event.x)
        canvas_y = self.canvas.canvasy(event.y)
        
        # Convert to image coordinates
        if self.scale_factor > 0:
            canvas_width = self.canvas.winfo_width()
            canvas_height = self.canvas.winfo_height()
            
            # Account for centering
            img_display_width = int(self.current_frame.shape[1] * self.scale_factor)
            img_display_height = int(self.current_frame.shape[0] * self.scale_factor)
            
            offset_x = (canvas_width - img_display_width) // 2
            offset_y = (canvas_height - img_display_height) // 2
            
            img_x = (canvas_x - offset_x) / self.scale_factor
            img_y = (canvas_y - offset_y) / self.scale_factor
            
            # Ensure coordinates are within image bounds
            img_height, img_width = self.current_frame.shape[:2]
            if 0 <= img_x < img_width and 0 <= img_y < img_height:
                self.click_points.append((img_x, img_y, is_positive, self.current_object_id))
                self.update_points_display()
                self.display_current_frame()
                
    def update_points_display(self):
        """Update the points display label"""
        if self.click_points:
            # Count points by object
            object_counts = {}
            for _, _, is_pos, obj_id in self.click_points:
                if obj_id not in object_counts:
                    object_counts[obj_id] = {'pos': 0, 'neg': 0}
                if is_pos:
                    object_counts[obj_id]['pos'] += 1
                else:
                    object_counts[obj_id]['neg'] += 1
            
            # Create summary text
            total_points = len(self.click_points)
            current_obj_points = object_counts.get(self.current_object_id, {'pos': 0, 'neg': 0})
            points_text = f"Total: {total_points} | Obj{self.current_object_id}: +{current_obj_points['pos']}, -{current_obj_points['neg']}"
        else:
            points_text = "None (Left click: +, Right click: -)"
        
        self.points_label.config(text=points_text)
            
    def clear_points(self):
        """Clear all click points"""
        self.click_points = []
        self.update_points_display()
        if self.frames:
            self.display_current_frame()
    
    def clear_current_object(self):
        """Clear points and masks for current object only"""
        # Remove points for current object
        self.click_points = [p for p in self.click_points if p[3] != self.current_object_id]
        
        # Remove masks for current object
        for frame_idx in self.masks:
            if self.current_object_id in self.masks[frame_idx]:
                del self.masks[frame_idx][self.current_object_id]
        
        self.update_points_display()
        if self.frames:
            self.display_current_frame()
            
    def toggle_mask_display(self):
        """Toggle mask overlay display"""
        if self.frames:
            self.display_current_frame()
            
    def prev_frame(self):
        """Go to previous frame"""
        if self.frames and self.current_frame_idx > 0:
            self.current_frame_idx -= 1
            self.display_current_frame()
            
    def next_frame(self):
        """Go to next frame"""
        if self.frames and self.current_frame_idx < len(self.frames) - 1:
            self.current_frame_idx += 1
            self.display_current_frame()
            
    def reset_video(self):
        """Reset to first frame"""
        if self.frames:
            self.current_frame_idx = 0
            self.display_current_frame()
            
    def on_slider_change(self, value):
        """Handle frame slider change"""
        if self.frames and not self.playing:
            self.current_frame_idx = int(float(value))
            self.display_current_frame()
            
    def toggle_play(self):
        """Toggle video playback"""
        if not self.frames:
            return
            
        self.playing = not self.playing
        if self.playing:
            self.play_button.config(text="⏸️ Pause")
            threading.Thread(target=self.play_video, daemon=True).start()
        else:
            self.play_button.config(text="▶️ Play")
            
    def play_video(self):
        """Play video in separate thread"""
        while self.playing and self.frames:
            if self.current_frame_idx < len(self.frames) - 1:
                self.current_frame_idx += 1
                self.root.after(0, self.display_current_frame)
                threading.Event().wait(0.033)  # ~30 FPS
            else:
                self.playing = False
                self.root.after(0, lambda: self.play_button.config(text="▶️ Play"))
                break
                
    def segment_video(self):
        """Perform SAM2 video segmentation"""
        if not self.frames:
            messagebox.showwarning("Warning", "Please load a video first")
            return
            
        if not self.model_loaded or not self.sam2_model:
            messagebox.showwarning("Warning", "Please load SAM2 model first")
            return
            
        if not self.click_points:
            messagebox.showwarning("Warning", "Please add some click points first")
            return
            
        try:
            self.status_label.config(text="Preparing frames for SAM2...")
            self.progress_bar.pack(fill=tk.X, pady=(5, 0))
            self.progress_var.set(0)
            self.root.update()
            
            # Create temporary directory for frames
            temp_dir = tempfile.mkdtemp(prefix='sam2_frames_')
            
            try:
                # Save all frames as JPEG files in the temporary directory
                for frame_idx, frame in enumerate(self.frames):
                    # Convert RGB to BGR for OpenCV
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    frame_path = os.path.join(temp_dir, f"{frame_idx:05d}.jpg")
                    cv2.imwrite(frame_path, frame_bgr)
                    
                    progress = (frame_idx / len(self.frames)) * 30  # 30% for frame saving
                    self.progress_var.set(progress)
                    if frame_idx % 10 == 0:
                        self.root.update_idletasks()
                
                self.status_label.config(text="Initializing SAM2 inference...")
                self.progress_var.set(35)
                self.root.update()
                
                # Initialize inference state with the temporary directory
                self.inference_state = self.sam2_model.init_state(video_path=temp_dir)
                
                # Group click points by object ID
                points_by_object = {}
                for x, y, is_pos, obj_id in self.click_points:
                    if obj_id not in points_by_object:
                        points_by_object[obj_id] = {'points': [], 'labels': []}
                    points_by_object[obj_id]['points'].append([x, y])
                    points_by_object[obj_id]['labels'].append(1 if is_pos else 0)
                
                self.status_label.config(text="Adding prompts to SAM2...")
                self.progress_var.set(40)
                self.root.update()
                
                # Initialize masks dictionary
                self.masks = {}
                for frame_idx in range(len(self.frames)):
                    self.masks[frame_idx] = {}
                
                # Process each object separately
                ann_frame_idx = self.current_frame_idx  # Use current frame as annotation frame
                
                for obj_id, point_data in points_by_object.items():
                    points = np.array(point_data['points'], dtype=np.float32)
                    labels = np.array(point_data['labels'], dtype=np.int32)
                    
                    self.status_label.config(text=f"Processing object {obj_id}...")
                    self.root.update()
                    
                    # Add new points for this object
                    try:
                        # Try the standard call first
                        result = self.sam2_model.add_new_points(
                            inference_state=self.inference_state,
                            frame_idx=ann_frame_idx,
                            obj_id=obj_id,
                            points=points,
                            labels=labels,
                        )
                        
                        # Handle different possible return formats
                        if isinstance(result, tuple):
                            if len(result) == 3:
                                _, out_obj_ids, out_mask_logits = result
                            elif len(result) == 2:
                                out_obj_ids, out_mask_logits = result
                            else:
                                # Fallback - assume it's just mask logits
                                out_mask_logits = result[0] if len(result) > 0 else result
                                out_obj_ids = [obj_id]
                        else:
                            # Single return value
                            out_mask_logits = result
                            out_obj_ids = [obj_id]
                            
                    except Exception as e:
                        print(f"Error adding points for object {obj_id}: {e}")
                        continue
                
                self.status_label.config(text="Propagating segmentation through video...")
                self.progress_var.set(45)
                self.root.update()
                
                # Propagate through entire video
                processed_frames = 0
                
                try:
                    # Get all video segments by propagating through the video
                    for out_frame_idx, out_obj_ids, out_mask_logits in self.sam2_model.propagate_in_video(self.inference_state):
                        # Process each object mask
                        for i, out_obj_id in enumerate(out_obj_ids):
                            if out_obj_id in points_by_object:  # Only process objects we have points for
                                # Convert logits to binary mask
                                mask_logits = out_mask_logits[i]
                                
                                # Handle torch tensors
                                if hasattr(mask_logits, 'cpu'):
                                    mask_logits = mask_logits.cpu()
                                if hasattr(mask_logits, 'numpy'):
                                    mask_logits = mask_logits.numpy()
                                
                                # Convert to binary mask
                                mask = (mask_logits > 0.0)
                                
                                # Ensure mask is 2D and convert to uint8
                                if len(mask.shape) > 2:
                                    mask = mask.squeeze()
                                
                                # Store mask for this object and frame
                                self.masks[out_frame_idx][out_obj_id] = (mask * 255).astype(np.uint8)
                        
                        processed_frames += 1
                        progress = 45 + (processed_frames / len(self.frames)) * 55  # 55% for processing
                        self.progress_var.set(min(progress, 100))
                        
                        if processed_frames % 10 == 0:  # Update every 10 frames
                            self.status_label.config(text=f"Processing frame {processed_frames}/{len(self.frames)}")
                            self.root.update_idletasks()
                            
                except Exception as e:
                    print(f"Error during propagation: {e}")
                    # Try alternative approach if propagation fails
                    messagebox.showwarning("Propagation Warning", 
                                         f"Video propagation encountered an issue: {str(e)}\n\n"
                                         f"Only the current frame will be segmented.")
                    
                    # Fallback: segment only current frame
                    for obj_id in points_by_object.keys():
                        if hasattr(self, 'out_mask_logits'):  # If we got masks from add_new_points
                            mask = (self.out_mask_logits > 0.0)
                            if hasattr(mask, 'cpu'):
                                mask = mask.cpu().numpy()
                            if len(mask.shape) > 2:
                                mask = mask.squeeze()
                            self.masks[ann_frame_idx][obj_id] = (mask * 255).astype(np.uint8)
                
                self.progress_bar.pack_forget()
                
                # Count total masks generated
                total_masks = sum(len(frame_masks) for frame_masks in self.masks.values())
                
                if total_masks > 0:
                    self.status_label.config(text=f"Segmentation complete! Generated {total_masks} masks")
                    
                    # Enable mask display
                    self.show_masks_var.set(True)
                    self.display_current_frame()
                    
                    messagebox.showinfo("Success", 
                                      f"Video segmentation completed!\n"
                                      f"Input: {len(self.frames)} frames\n"
                                      f"Objects: {len(points_by_object)}\n"
                                      f"Generated: {total_masks} masks\n\n"
                                      f"Mask overlay is now enabled.")
                else:
                    self.status_label.config(text="Segmentation completed but no masks generated")
                    messagebox.showwarning("Warning", 
                                         "Segmentation completed but no masks were generated. "
                                         "This might be due to:\n"
                                         "1. Points not being clear enough\n"
                                         "2. Model having difficulty with the content\n"
                                         "3. Technical issues with the model\n\n"
                                         "Try adding more/different points or restarting.")
                
            finally:
                # Clean up temporary directory
                try:
                    shutil.rmtree(temp_dir)
                except Exception as e:
                    print(f"Warning: Could not clean up temporary directory {temp_dir}: {e}")
                
        except Exception as e:
            self.progress_bar.pack_forget()
            error_msg = f"Segmentation failed: {str(e)}\n\nThis might be due to:\n"
            error_msg += "1. GPU memory issues (try smaller model or fewer objects)\n"
            error_msg += "2. Incompatible video format\n" 
            error_msg += "3. SAM2 installation issues\n"
            error_msg += "4. Model checkpoint compatibility\n\n"
            error_msg += "Try restarting the application or using a smaller video."
            self.status_label.config(text="Segmentation failed")
            messagebox.showerror("Segmentation Error", error_msg)
            
        
    def export_masks(self):
        """Export segmentation masks"""
        if not self.masks:
            messagebox.showwarning("Warning", "No masks to export. Please segment the video first.")
            return
            
        folder_path = filedialog.askdirectory(title="Select folder to save masks")
        if not folder_path:
            return
            
        try:
            self.status_label.config(text="Exporting masks...")
            self.progress_bar.pack(fill=tk.X, pady=(5, 0))
            self.root.update()
            
            # Export masks as PNG files
            exported_count = 0
            for frame_idx, frame_masks in self.masks.items():
                for obj_id, mask in frame_masks.items():
                    mask_path = os.path.join(folder_path, f"mask_frame_{frame_idx:05d}_obj_{obj_id}.png")
                    cv2.imwrite(mask_path, mask)
                    exported_count += 1
                    
                    progress = (exported_count / sum(len(fm) for fm in self.masks.values())) * 90
                    self.progress_var.set(progress)
                    
                    if exported_count % 50 == 0:
                        self.root.update_idletasks()
            
            # Export metadata
            metadata = {
                "video_path": self.video_path,
                "total_frames": len(self.frames),
                "objects": {},
                "click_points_by_object": {},
                "prompt_frame": self.current_frame_idx,
                "export_timestamp": str(__import__('datetime').datetime.now()),
                "sam2_paths": {
                    "base_path": self.sam2_base_path,
                    "checkpoint_dir": self.checkpoint_dir,
                    "config_dir": self.config_dir
                }
            }
            
            # Group click points by object
            for x, y, is_pos, obj_id in self.click_points:
                if obj_id not in metadata["click_points_by_object"]:
                    metadata["click_points_by_object"][obj_id] = []
                metadata["click_points_by_object"][obj_id].append({
                    "x": float(x), "y": float(y), "positive": bool(is_pos)
                })
            
            # Count masks per object
            for frame_idx, frame_masks in self.masks.items():
                for obj_id in frame_masks.keys():
                    if obj_id not in metadata["objects"]:
                        metadata["objects"][obj_id] = {"mask_count": 0, "color": self.object_colors.get(obj_id, [255, 255, 255])}
                    metadata["objects"][obj_id]["mask_count"] += 1
            
            metadata_path = os.path.join(folder_path, "segmentation_metadata.json")
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            self.progress_var.set(100)
            self.root.update_idletasks()
            
            self.progress_bar.pack_forget()
            self.status_label.config(text=f"Export complete: {exported_count} masks saved")
            
            messagebox.showinfo("Export Complete", 
                              f"Successfully exported:\n"
                              f"• {exported_count} mask images (PNG)\n"
                              f"• 1 metadata file (JSON)\n\n"
                              f"Location: {folder_path}")
                
        except Exception as e:
            self.progress_bar.pack_forget()
            self.status_label.config(text="Export failed")
            messagebox.showerror("Export Error", f"Failed to export masks: {str(e)}")

    def export_video(self):
        """Export segmented video with various options - FIXED VERSION"""
        if not self.masks:
            messagebox.showwarning("Warning", "No masks to export. Please segment the video first.")
            return
            
        if not self.frames:
            messagebox.showwarning("Warning", "No video frames available.")
            return
        
        # Create export options dialog
        export_dialog = tk.Toplevel(self.root)
        export_dialog.title("Export Video Options")
        export_dialog.geometry("450x500")  # Increased height to fit all elements
        export_dialog.configure(bg='#2b2b2b')
        export_dialog.transient(self.root)
        export_dialog.grab_set()
        
        # Center the dialog
        export_dialog.update_idletasks()
        x = (export_dialog.winfo_screenwidth() // 2) - (export_dialog.winfo_width() // 2)
        y = (export_dialog.winfo_screenheight() // 2) - (export_dialog.winfo_height() // 2)
        export_dialog.geometry(f"+{x}+{y}")
        
        # Configure dialog style
        dialog_style = ttk.Style()
        dialog_style.theme_use('clam')
        
        # Configure colors
        dialog_style.configure('Dialog.TFrame', background='#2b2b2b')
        dialog_style.configure('Dialog.TLabel', background='#2b2b2b', foreground='white')
        dialog_style.configure('Dialog.TButton', background='#404040', foreground='white')
        dialog_style.map('Dialog.TButton', 
                         background=[('active', '#505050'), ('pressed', '#303030')])
        
        main_frame = ttk.Frame(export_dialog, style='Dialog.TFrame')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Title
        title_label = ttk.Label(main_frame, text="Video Export Options", 
                               font=('Arial', 14, 'bold'), style='Dialog.TLabel')
        title_label.pack(pady=(0, 20))
        
        # Export type selection
        export_type_var = tk.StringVar(value="overlay")
        
        ttk.Label(main_frame, text="Export Type:", font=('Arial', 10, 'bold'), 
                 style='Dialog.TLabel').pack(anchor=tk.W, pady=(0, 5))
        
        type_frame = ttk.Frame(main_frame, style='Dialog.TFrame')
        type_frame.pack(fill=tk.X, pady=(0, 15))
        
        ttk.Radiobutton(type_frame, text="Original with mask overlay", 
                       variable=export_type_var, value="overlay").pack(anchor=tk.W, pady=2)
        ttk.Radiobutton(type_frame, text="Mask only (black background)", 
                       variable=export_type_var, value="mask_only").pack(anchor=tk.W, pady=2)
        ttk.Radiobutton(type_frame, text="Segmented object only", 
                       variable=export_type_var, value="object_only").pack(anchor=tk.W, pady=2)
        ttk.Radiobutton(type_frame, text="Side-by-side comparison", 
                       variable=export_type_var, value="side_by_side").pack(anchor=tk.W, pady=2)
        
        # Quality settings
        ttk.Label(main_frame, text="Video Quality:", font=('Arial', 10, 'bold'), 
                 style='Dialog.TLabel').pack(anchor=tk.W, pady=(10, 5))
        
        quality_frame = ttk.Frame(main_frame, style='Dialog.TFrame')
        quality_frame.pack(fill=tk.X, pady=(0, 15))
        
        fps_var = tk.DoubleVar(value=30.0)
        ttk.Label(quality_frame, text="FPS:", style='Dialog.TLabel').pack(side=tk.LEFT)
        fps_spinbox = tk.Spinbox(quality_frame, from_=1, to=60, textvariable=fps_var, 
                                width=8, bg='#404040', fg='white', insertbackground='white')
        fps_spinbox.pack(side=tk.LEFT, padx=(5, 20))
        
        # Overlay transparency
        ttk.Label(main_frame, text="Overlay Settings:", font=('Arial', 10, 'bold'), 
                 style='Dialog.TLabel').pack(anchor=tk.W, pady=(10, 5))
        
        overlay_frame = ttk.Frame(main_frame, style='Dialog.TFrame')
        overlay_frame.pack(fill=tk.X, pady=(0, 15))
        
        overlay_alpha_var = tk.DoubleVar(value=0.4)
        ttk.Label(overlay_frame, text="Transparency:", style='Dialog.TLabel').pack(anchor=tk.W)
        alpha_scale = ttk.Scale(overlay_frame, from_=0.1, to=0.8, variable=overlay_alpha_var, 
                               orient=tk.HORIZONTAL)
        alpha_scale.pack(fill=tk.X, pady=5)
        
        # Object selection for export
        ttk.Label(main_frame, text="Objects to Export:", font=('Arial', 10, 'bold'), 
                 style='Dialog.TLabel').pack(anchor=tk.W, pady=(10, 5))
        
        # Create scrollable frame for objects
        objects_container = ttk.Frame(main_frame, style='Dialog.TFrame')
        objects_container.pack(fill=tk.X, pady=(0, 20))
        
        # Get unique object IDs from masks
        unique_objects = set()
        for frame_masks in self.masks.values():
            unique_objects.update(frame_masks.keys())
        unique_objects = sorted(list(unique_objects))
        
        export_objects_vars = {}
        if unique_objects:
            for obj_id in unique_objects:
                export_objects_vars[obj_id] = tk.BooleanVar(value=True)
                color_hex = self._rgb_to_hex(self.object_colors.get(obj_id, [255, 255, 255]))
                
                obj_frame = ttk.Frame(objects_container, style='Dialog.TFrame')
                obj_frame.pack(anchor=tk.W, pady=1)
                
                cb = ttk.Checkbutton(obj_frame, text=f"Object {obj_id}", 
                                   variable=export_objects_vars[obj_id])
                cb.pack(side=tk.LEFT)
                
                # Color indicator
                color_label = ttk.Label(obj_frame, text="●", foreground=color_hex, 
                                       font=('Arial', 12), style='Dialog.TLabel')
                color_label.pack(side=tk.LEFT, padx=(5, 0))
        else:
            # If no objects found in masks, show message
            ttk.Label(objects_container, text="No segmented objects found.", 
                     style='Dialog.TLabel', foreground='orange').pack(anchor=tk.W)
            # Still need to initialize the dict for the buttons to work
            export_objects_vars = {}
        
        # Buttons frame
        button_frame = ttk.Frame(main_frame, style='Dialog.TFrame')
        button_frame.pack(fill=tk.X, pady=(20, 0))
        
        def start_export():
            # Get selected objects
            selected_objects = [obj_id for obj_id, var in export_objects_vars.items() if var.get()]
            if not selected_objects and export_objects_vars:
                messagebox.showwarning("Warning", "Please select at least one object to export.")
                return
            
            # If no objects in masks, export with empty list (will show original frames)
            if not export_objects_vars:
                selected_objects = []
            
            export_dialog.destroy()
            self._export_video_with_options(
                export_type_var.get(),
                fps_var.get(),
                overlay_alpha_var.get(),
                selected_objects
            )
        
        def cancel_export():
            export_dialog.destroy()
        
        # Create buttons with proper styling and sizing
        cancel_btn = ttk.Button(button_frame, text="Cancel", command=cancel_export, 
                               width=12, style='Dialog.TButton')
        cancel_btn.pack(side=tk.RIGHT, padx=(5, 0))
        
        export_btn = ttk.Button(button_frame, text="Export Video", command=start_export, 
                               width=15, style='Dialog.TButton')
        export_btn.pack(side=tk.RIGHT, padx=(5, 5))
        
        # Configure accent button style for export button
        try:
            dialog_style.configure('Accent.TButton', 
                                 background='#0078d4', 
                                 foreground='white',
                                 font=('Arial', 9, 'bold'))
            dialog_style.map('Accent.TButton',
                           background=[('active', '#106ebe'), ('pressed', '#005a9e')])
            export_btn.configure(style='Accent.TButton')
        except:
            pass
        
        # Make export button the default (activated by Enter key)
        export_dialog.bind('<Return>', lambda e: start_export())
        export_dialog.bind('<Escape>', lambda e: cancel_export())
        
        # Focus on export button
        export_btn.focus_set()

    def _export_video_with_options(self, export_type, fps, overlay_alpha, selected_objects):
        """Export video with specified options - FIXED AND COMPLETE VERSION"""
        # Ask for output file
        output_path = filedialog.asksaveasfilename(
            title="Save Video As",
            defaultextension=".mp4",
            filetypes=[
                ("MP4 files", "*.mp4"),
                ("AVI files", "*.avi"),
                ("MOV files", "*.mov"),
                ("All files", "*.*")
            ]
        )
        
        if not output_path:
            return
            
        try:
            self.status_label.config(text="Preparing video export...")
            self.progress_bar.pack(fill=tk.X, pady=(5, 0))
            self.progress_var.set(0)
            self.root.update()
            
            # Get video properties
            if self.frames:
                height, width = self.frames[0].shape[:2]
                
                # Adjust dimensions based on export type
                if export_type == "side_by_side":
                    output_width = width * 2
                    output_height = height
                else:
                    output_width = width
                    output_height = height
                
                # Initialize video writer
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(output_path, fourcc, fps, (output_width, output_height))
                
                if not out.isOpened():
                    raise ValueError("Could not open video writer. Try a different output path or format.")
                
                total_frames = len(self.frames)
                processed_frames = 0
                
                self.status_label.config(text="Exporting video frames...")
                self.root.update()
                
                for frame_idx, frame in enumerate(self.frames):
                    # Get masks for this frame if available
                    masks_dict = {}
                    if frame_idx in self.masks:
                        # Filter masks by selected objects
                        for obj_id, mask in self.masks[frame_idx].items():
                            if obj_id in selected_objects:
                                masks_dict[obj_id] = mask
                    
                    # Create output frame based on export type
                    if export_type == "overlay":
                        output_frame = self._create_overlay_frame(frame, masks_dict, overlay_alpha)
                    elif export_type == "mask_only":
                        output_frame = self._create_mask_only_frame(frame, masks_dict)
                    elif export_type == "object_only":
                        output_frame = self._create_object_only_frame(frame, masks_dict)
                    elif export_type == "side_by_side":
                        output_frame = self._create_side_by_side_frame(frame, masks_dict, overlay_alpha)
                    else:
                        output_frame = frame.copy()
                    
                    # Convert RGB to BGR for OpenCV
                    output_frame_bgr = cv2.cvtColor(output_frame, cv2.COLOR_RGB2BGR)
                    out.write(output_frame_bgr)
                    
                    processed_frames += 1
                    progress = (processed_frames / total_frames) * 100
                    self.progress_var.set(progress)
                    
                    if processed_frames % 30 == 0:  # Update every 30 frames
                        self.status_label.config(text=f"Exporting frame {processed_frames}/{total_frames}")
                        self.root.update_idletasks()
                
                out.release()
                
                self.progress_bar.pack_forget()
                self.status_label.config(text=f"Video export complete!")
                
                messagebox.showinfo("Export Complete", 
                                  f"Video exported successfully!\n\n"
                                  f"Type: {export_type.replace('_', ' ').title()}\n"
                                  f"Frames: {total_frames}\n"
                                  f"FPS: {fps}\n"
                                  f"Objects: {len(selected_objects)}\n"
                                  f"Location: {output_path}")
                
        except Exception as e:
            self.progress_bar.pack_forget()
            self.status_label.config(text="Video export failed")
            messagebox.showerror("Export Error", f"Failed to export video: {str(e)}")

    def _create_overlay_frame(self, frame, masks_dict, alpha):
        """Create frame with mask overlay - Updated to handle multiple objects"""
        output_frame = frame.copy()
        if masks_dict:
            for obj_id, mask in masks_dict.items():
                # Use object-specific color
                obj_color = self.object_colors.get(obj_id, [0, 255, 255])
                overlay = np.zeros_like(frame)
                overlay[mask > 0] = obj_color
                output_frame = cv2.addWeighted(output_frame, 1-alpha, overlay, alpha, 0)
        return output_frame

    def _create_mask_only_frame(self, frame, masks_dict):
        """Create frame showing only the masks - Updated for multiple objects"""
        height, width = frame.shape[:2]
        output_frame = np.zeros((height, width, 3), dtype=np.uint8)
        if masks_dict:
            for obj_id, mask in masks_dict.items():
                obj_color = self.object_colors.get(obj_id, [255, 255, 255])
                output_frame[mask > 0] = obj_color
        return output_frame

    def _create_object_only_frame(self, frame, masks_dict):
        """Create frame showing only the segmented objects - Updated for multiple objects"""
        output_frame = np.zeros_like(frame)
        if masks_dict:
            combined_mask = np.zeros(frame.shape[:2], dtype=bool)
            for obj_id, mask in masks_dict.items():
                combined_mask = np.logical_or(combined_mask, mask > 0)
            
            # Apply combined mask to frame
            for c in range(3):
                output_frame[:, :, c] = np.where(combined_mask, frame[:, :, c], 0)
        return output_frame

    def _create_side_by_side_frame(self, frame, masks_dict, alpha):
        """Create side-by-side comparison frame - Updated for multiple objects"""
        # Original frame on the left
        left_frame = frame.copy()
        
        # Overlay frame on the right
        right_frame = self._create_overlay_frame(frame, masks_dict, alpha)
        
        # Combine side by side
        output_frame = np.hstack([left_frame, right_frame])
        return output_frame

def main():
    """Main application entry point"""
    root = tk.Tk()
    
    # Set window icon if available
    try:
        # You can add an icon file here if you have one
        # root.iconbitmap('icon.ico')
        pass
    except:
        pass
    
    # Center window on screen
    root.update_idletasks()
    width = root.winfo_width()
    height = root.winfo_height()
    x = (root.winfo_screenwidth() // 2) - (width // 2)
    y = (root.winfo_screenheight() // 2) - (height // 2)
    root.geometry(f'{width}x{height}+{x}+{y}')
    
    # Create and run application
    app = SAM2VideoUI(root)
    
    # Handle window closing
    def on_closing():
        if messagebox.askokcancel("Quit", "Do you want to quit the application?"):
            root.destroy()
    
    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()

if __name__ == "__main__":
    main()