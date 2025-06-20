import cv2
import torch
import csv
import os

def main(video_path):
    # Load the YOLOv5 model (small version pretrained on COCO)
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Prepare output CSV file path
    output_dir = os.path.join(os.getcwd(), 'output')
    os.makedirs(output_dir, exist_ok=True)  
    csv_filepath = os.path.join(output_dir, 'person_count_results.csv')

    with open(csv_filepath, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['frame_number', 'person_count']) 

        frame_num = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break  # end of video

            # Run detection on frame
            results = model(frame)

            # Extract detected objects as pandas dataframe
            detections = results.pandas().xyxy[0]

            # Filter detections for only 'person' class
            persons = detections[detections['name'] == 'person']
            person_count = len(persons)

            # Draw bounding boxes around people
            for _, row in persons.iterrows():
                x1, y1, x2, y2 = map(int, [row['xmin'], row['ymin'], row['xmax'], row['ymax']])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, 'Person', (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Display person count on frame
            cv2.putText(frame, f'Persons: {person_count}', (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # Show the frame
            cv2.imshow('Person Detection', frame)

            # Save data to CSV
            writer.writerow([frame_num, person_count])

            # Break if 'q' key is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            frame_num += 1

    cap.release()
    cv2.destroyAllWindows()
    print(f'Detection complete! Results saved to: {csv_filepath}')


if __name__ == '__main__':
    video_path = r"E:\crowd_detection\input\input_video.mp4"  
    main(video_path)
