from ultralytics import YOLO
import cv2

# Load your trained YOLOv8 model
def tester(model_path, video_input_path, video_output_path):
    model = YOLO(model_path)  # Replace with your model path

    # Path to the input video
    input_video_path = video_input_path
    output_video_path = video_output_path

    # Open the video file

    cap = cv2.VideoCapture(input_video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))


    # Define the codec and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 files
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # Process each frame
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break


        # Run YOLOv8 inference on the frame
        results = model(frame)

        # Draw the detection results on the frame
        annotated_frame = results[0].plot(labels=False)  # Annotated frame with boxes and labels

        # Write the frame to the output video
        out.write(annotated_frame)

    # Release resources

    cap.release()
    out.release()

    print(f"Inference complete! Saved to {output_video_path}")

if __name__ == '__main__':
    model = 'path to whatever model you just trained was'

    video_in = 'video u tryna train apon'
    video_out = 'this drone shit'
    tester(model, video_in, video_out)
