from imports import *
# Load the pre-trained YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Set the model in evaluation mode
model.eval()

fall_model = joblib.load("./detect_fall.joblib")
base_options_pos = python.BaseOptions(model_asset_path='pose_landmarker_full.task')
options_pos = vision.PoseLandmarkerOptions(
    base_options=base_options_pos,
    output_segmentation_masks=True)
detector_pos = vision.PoseLandmarker.create_from_options(options_pos)

def extract_position_on_image(rgb_image ,detector,model):
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.array(rgb_image))
    detection_result = detector.detect(image)
    try:
        body_positions = np.array(detection_result.pose_landmarks[0])

        # Append the image filename and body positions to the data list
        if body_positions is not None:
            row = {}
            for i in range(len(body_positions)):
                row["x"+str(i)] = body_positions[i].x
                row["y"+str(i)] = body_positions[i].y
                row["z"+str(i)] = body_positions[i].z
            data = [row]
        df2 = pd.DataFrame(data)
        prediction = model.predict(df2)
        return prediction[0], draw_landmarks_on_image(rgb_image, detection_result)
    except:
        -1, draw_landmarks_on_image(rgb_image, detection_result)
def draw_landmarks_on_image(rgb_image, detection_result):
  pose_landmarks_list = detection_result.pose_landmarks
  annotated_image = np.copy(rgb_image)

  # Loop through the detected poses to visualize.
  for idx in range(len(pose_landmarks_list)):
    pose_landmarks = pose_landmarks_list[idx]

    # Draw the pose landmarks.
    pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    pose_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
    ])
    solutions.drawing_utils.draw_landmarks(
      annotated_image,
      pose_landmarks_proto,
      solutions.pose.POSE_CONNECTIONS,
      solutions.drawing_styles.get_default_pose_landmarks_style())
  return annotated_image



def draw(frame):
    er = DetectFace(device='cpu', gpu_id=0)
    pre = {-1:"unkown", 1: "standing", 0: "falling", 2:"sitting"}
    TEXT_COLOR = (255, 0, 0)
    MARGIN = 10
    MARGIN = 10  # pixels
    ROW_SIZE = 10  # pixels
    FONT_SIZE = 1
    FONT_THICKNESS = 3
    results = model(frame)

    # Retrieve the detected objects
    detected_objects = results.xyxy[0]  # Index 0 represents the first detected object
    
    # Iterate over the detected objects and filter out human bodies
    for detection in detected_objects:
        class_label = detection[-1]
        if class_label == 0:  # 0 represents the class label for humans
            x1, y1, x2, y2, confidence, _ = detection
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            frame_person = frame[y1:y2, x1:x2]
            frame_person, emotion = er.predict_emotion(frame_person)
            try:
                pred, frame_person = extract_position_on_image(frame_person, detector_pos,fall_model)
            except:
                pred = -1
            frame[y1:y2, x1:x2] = frame_person
            text_location = (MARGIN + x1,
                     MARGIN + ROW_SIZE + y1)
            cv.putText(frame, str(pre[pred]), text_location, cv.FONT_HERSHEY_PLAIN, FONT_SIZE, TEXT_COLOR, FONT_THICKNESS)
    return frame

# Create the root window
root = tk.Tk()
root.withdraw()  # Hide the root window

# Open the file explorer dialog
input_path = filedialog.askopenfilename()

# Destroy the root window
root.destroy()
# Check if a file was selected
if input_path:
    print("Selected file:", input_path)
    _, file_extension = os.path.splitext(input_path)
    output_path = os.path.splitext(input_path)[0]+'_output'+ file_extension
    if file_extension.lower() in ['.jpg', '.png', '.bmp']:
        image = draw(cv.imread(input_path))
        cv.imwrite(output_path,image)
        print(f"your file is ready in {output_path}")
        cv.imshow("output", image)
        cv.waitKey(0)
    elif file_extension.lower() in ['.mp4', '.avi', '.mov']:
        # Read in the video file
        video = cv.VideoCapture(input_path)

        # Get the frames per second (fps) of the video
        fps = int(video.get(cv.CAP_PROP_FPS))

        width = int(video.get(cv.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(video.get(cv.CAP_PROP_FRAME_COUNT))
        writer = cv.VideoWriter(output_path, int(video.get(cv.CAP_PROP_FOURCC)), fps, (width, height))

        # Loop through the video frames
        with tqdm(total=total_frames, desc='Processing frames') as pbar:
            frame_count = 0
            while True:
                # Read in a single frame
                ret, frame = video.read()
                
                # Check if we have reached the end of the video
                if not ret:
                    break
                
                # Write the edited frame to the output file
                writer.write(draw(frame))
                pbar.update(1)
                frame_count += 1
                pbar.set_postfix({'Frame': frame_count})
            
        # Release the resources used by the VideoCapture and VideoWriter objects
        video.release()
        writer.release()
        print(f"your file is ready in {output_path}")
    else:
        print('unkown input')
else:
    print("No file selected.")


