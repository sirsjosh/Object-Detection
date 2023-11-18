import cv2
from moviepy.editor import VideoFileClip
from lesson_functions import *
from scipy.ndimage.measurements import label

# Load the trained SVM model and scaler
svc = joblib.load('svm_model.pkl')
X_scaler = joblib.load('scaler.pkl')

# Function to process each frame in the video
def process_frame(frame):
    # Define parameters
    color_space = 'RGB'
    orient = 9
    pix_per_cell = 8
    cell_per_block = 2
    hog_channel = 0
    spatial_size = (16, 16)
    hist_bins = 16
    spatial_feat = True
    hist_feat = True
    hog_feat = True
    y_start_stop = [400, 656]

    # Create a copy of the frame to draw on
    draw_frame = np.copy(frame)

    # Scale the frame to 0-1 and convert to RGB if necessary
    frame = frame.astype(np.float32) / 255.0
    if np.max(frame) > 1.0:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Find windows with positive detections
    windows = slide_window(frame, x_start_stop=[None, None], y_start_stop=y_start_stop,
                            xy_window=(96, 96), xy_overlap=(0.5, 0.5))

    hot_windows = search_windows(frame, windows, svc, X_scaler, color_space=color_space,
                                 spatial_size=spatial_size, hist_bins=hist_bins,
                                 orient=orient, pix_per_cell=pix_per_cell,
                                 cell_per_block=cell_per_block,
                                 hog_channel=hog_channel, spatial_feat=spatial_feat,
                                 hist_feat=hist_feat, hog_feat=hog_feat)

    # Create a heatmap and threshold to remove false positives
    heat = np.zeros_like(frame[:, :, 0]).astype(np.float)
    heat = add_heat(heat, hot_windows)
    heat = apply_threshold(heat, 1)

    # Visualize the heatmap
    heatmap = np.clip(heat, 0, 255)

    # Find final boxes from heatmap using label function
    labels = label(heatmap)
    draw_frame = draw_labeled_bboxes(draw_frame, labels)

    return draw_frame

# Process the video
input_video = './project_video.mp4'
output_video = 'output_video.mp4'
clip = VideoFileClip(input_video)
output_clip = clip.fl_image(process_frame)
output_clip.write_videofile(output_video, audio=False)
