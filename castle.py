from tkinter import *
import cv2
import threading
import numpy as np

# Shared variables and thread lock
slider1_value = 50
lock = threading.Lock()

# Global variables to store saved target centroid and mask radius
saved_centroid = None
saved_two = None
mask_radius = 50  # Initial radius for the saved target mask

# Function to display the OpenCV video feed
def start_camera():
    global slider1_value, saved_centroid, saved_two, mask_radius, a
    a = 0
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Get frame dimensions
        height, width, _ = frame.shape
        center_x, center_y = width // 2, height // 2

        # Read slider value safely
        with lock:
            radius = slider1_value
            mask_radius = slider1_value
        # Draw crosshairs and circle
        cv2.line(frame, (center_x, center_y - 10), (center_x, center_y + 10), (0, 128, 0), 2)
        cv2.line(frame, (center_x - 10, center_y), (center_x + 10, center_y), (0, 128, 0), 2)
        cv2.circle(frame, (center_x, center_y), radius, (0, 128, 0), 2)

        # Create a mask for black regions
        target_lo = np.array([0, 0, 0])  # Example lower range for black in RGB
        target_hi = np.array([50, 50, 50])  # Example upper range for black in RGB
        color_mask = cv2.inRange(frame, target_lo, target_hi)

        # Create a circular mask for the targeting circle
        y, x = np.ogrid[:height, :width]
        distance_from_center = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
        circular_mask = distance_from_center <= radius

        # Combine the color and circular masks
        combined_mask = np.bitwise_and(color_mask > 0, circular_mask)

        # Highlight the masked area in red
        frame[combined_mask] = [0, 0, 255]

        # Find the centroid of the masked region
        coords = cv2.findNonZero(combined_mask.astype(np.uint8))
        current_centroid = None
        if coords is not None:
            mean_coords = np.mean(coords, axis=0).astype(int)
            current_centroid = (mean_coords[0][0], mean_coords[0][1])
            cv2.circle(frame, current_centroid, 5, (255, 0, 0), -1)  # Draw current centroid

        # Handle the saved centroid and dynamic mask for ONE
        if saved_centroid:
            # Define a circular mask around the saved centroid
            saved_x, saved_y = saved_centroid
            distance_from_saved = np.sqrt((x - saved_x) ** 2 + (y - saved_y) ** 2)
            saved_mask = distance_from_saved <= mask_radius

            # Update the saved centroid based on the saved mask
            saved_coords = cv2.findNonZero(np.bitwise_and(color_mask > 0, saved_mask).astype(np.uint8))
            if saved_coords is not None:
                mean_saved_coords = np.mean(saved_coords, axis=0).astype(int)
                saved_centroid = (mean_saved_coords[0][0], mean_saved_coords[0][1])

            # Highlight the saved target in yellow
            cv2.circle(frame, saved_centroid, 10, (0, 255, 255), -1)
            cv2.circle(frame, saved_centroid, mask_radius, (0, 128, 0), 2)
            cv2.putText(frame, "ALPHA", (saved_centroid[0] + 15, saved_centroid[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            
        if saved_two:
            # Define a circular mask around the second saved centroid
            saved_xx, saved_yy = saved_two
            distance_from_saved2 = np.sqrt((x - saved_xx) ** 2 + (y - saved_yy) ** 2)
            saved_mask2 = distance_from_saved2 <= mask_radius

            # Update the second saved centroid based on the saved mask
            saved_coords2 = cv2.findNonZero(np.bitwise_and(color_mask > 0, saved_mask2).astype(np.uint8))
            if saved_coords2 is not None:
                mean_saved_coords2 = np.mean(saved_coords2, axis=0).astype(int)
                saved_two = (mean_saved_coords2[0][0], mean_saved_coords2[0][1])

            # Highlight the second saved target in cyan
            cv2.circle(frame, saved_two, 10, (255, 255, 0), -1)
            cv2.circle(frame, saved_two, mask_radius, (255, 255, 0), 2)
            cv2.putText(frame, "BETA", (saved_two[0] + 15, saved_two[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)


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
master = Tk()
w1 = Scale(master, from_=0, to=130, label="Size", command=update_slider_values)
w1.set(slider1_value)
w1.pack()

Button(master, text='Show Values', command=lambda: print(w1.get())).pack()

# Run the OpenCV video feed in a separate thread
camera_thread = threading.Thread(target=start_camera, daemon=True)
camera_thread.start()

# Handle Tkinter window close
def on_closing():
    master.destroy()
    camera_thread.join()

master.protocol("WM_DELETE_WINDOW", on_closing)
master.mainloop()
