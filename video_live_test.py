from tkinter import *
import cv2
import threading
import numpy as np
from ultralytics import YOLO

# Shared variables and thread lock
slider1_value = 50
lock = threading.Lock()

# Global variables to store saved target centroid and mask radius

saved_centroid = None
mask_radius = 50  # Initial radius for the saved target mask

# Function to display the OpenCV video feed
def start_camera(model):
    global slider1_value, saved_centroid, a
    a = 0
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break


        # Get frame dimensions
        height, width, _ = frame.shape

        results = model(frame)

        if results is not None:
            for box_label in results[0].boxes.xyxy:
                current_centroids = box_label[:4].cpu().numpy()
        
        current_centroid = [int((current_centroids[0][0] + current_centroids[0][2])/2), int((current_centroids[0][1] + current_centroids[0][3])/2)]
        avg_radius = int(np.mean(np.abs((current_centroids[0][0] - current_centroids[0][2])/2), np.abs((current_centroids[0][1] - current_centroids[0][3])/2)))

        if not current_centroid == None:
            saved_centroid = current_centroid

        if not saved_centroid == None:
            center_x, center_y = saved_centroid[0], saved_centroid[1]
        else:
            center_x, center_y = width//2, height//2

        # Read slider value safely
        with lock:
            radius = slider1_value
            mask_radius = slider1_value
        # Draw crosshairs and circle
        cv2.line(frame, (width//2, height//2), (center_x, center_y), (0, 150, 0), 2)
        cv2.circle(frame, (width//2, height//2), np.linalg.norm((width//2, height//2), (center_x, center_y)) * .1, (0 ,150, 0), 2)
        cv2.circle(frame, (center_x, center_y), avg_radius, (0, 150, 0), 2)

        cv2.putText(frame, "Drone", (center_x, center_y+10),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        
        

        
        
        # Handle the saved centroid and dynamic mask for ONE
        if saved_centroid:
            # Define a circular mask around the saved centroid
            # Highlight the saved target in yellow
            cv2.circle(frame, saved_centroid, 10, (0, 255, 255), -1)
            cv2.circle(frame, saved_centroid, mask_radius, (0, 128, 0), 2)
            cv2.putText(frame, "ALPHA", (saved_centroid[0] + 15, saved_centroid[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            
        

        # Check for keypress events
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):  # Stop the camera feed
            break
        elif key == ord('c') and current_centroid:  # Save the current centroid
            if a == 0:
                saved_centroid = current_centroid
            if a == 1:
                saved_two = current_centroid

            a += 1


        # Show the frame
        cv2.imshow('Camera Feed', frame)

    cap.release()
    cv2.destroyAllWindows()

# Function to update the slider value

def update_slider_values(val):
    global slider1_value
    with lock:
        slider1_value = int(val)

# Set up the Tkinter UI
def test(model):
    master = Tk()
    w1 = Scale(master, from_=0, to=130, label="Size", command=update_slider_values)
    w1.set(slider1_value)
    w1.pack()

    Button(master, text='Show Values', command=lambda: print(w1.get())).pack()

    # Run the OpenCV video feed in a separate thread

    camera_thread = threading.Thread(target=start_camera, args = (model,), daemon=True)
    camera_thread.start()

    # Handle Tkinter window close
    def on_closing():
        master.destroy()
        camera_thread.join()

    master.protocol("WM_DELETE_WINDOW", on_closing)
    master.mainloop()

if __name__ == '__main__':
    model = 'The path to the model you just trained and now its awesome are you happy?'
    test(model)
