import cv2
import numpy as np
from ultralytics import YOLO


def align_body_part(template, camera_image, seg_model):
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    _, template_thresh = cv2.threshold(template_gray, 127, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(template_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Initialize variables for alignment ratio and area color
    alignment_ratio = 0.0
    area_color = (0, 0, 255)  # Red color for misaligned body part
    answer = False

    # Create a copy of the camera image for displaying the mask
    mask_image = camera_image.copy()

    # Iterate over the contours (areas) in the template
    for contour in contours:
        # Get the bounding rectangle of the contour
        x, y, w, h = cv2.boundingRect(contour)

        # Extract the corresponding region from the camera image
        roi = camera_image[y:y + h, x:x + w]

        # Run YOLO segmentation on the ROI
        roi_results = seg_model(roi)

        # Check if any objects are detected in the ROI
        if len(roi_results) > 0 and roi_results[0].masks is not None:
            # Get the class labels and masks of the detected objects
            class_labels = roi_results[0].boxes.cls.cpu().numpy()
            masks = roi_results[0].masks.data

            # Resize the mask to match the size of the contour
            mask = masks[0].cpu().numpy()
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
            mask_contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # detour similarity
            template_moments = cv2.HuMoments(cv2.moments(contour)).flatten()
            mask_moments = cv2.HuMoments(cv2.moments(mask_contours[0])).flatten()
            similarity = cv2.matchShapes(template_moments, mask_moments, cv2.CONTOURS_MATCH_I1, 0)
            print("similarity:", similarity)

            # Calculate the ratio of the masked area to the contour area
            contour_area = cv2.contourArea(contour)
            mask_area = np.sum(mask)
            print("mask_area", mask_area)
            print("contour_area", contour_area)
            mask_ratio = mask_area / contour_area
            print("mask_ratio", mask_ratio)
            # Update the alignment ratio and area color if a better alignment is found
            if mask_ratio > threshold and 0 in class_labels and similarity > 85:
                area_color = (0, 255, 0)  # Green color for aligned body part
                answer = True

            # Create a mask image with light grey color
            mask_image_roi = np.zeros_like(roi)
            mask_image_roi[mask > 0] = (192, 192, 192)  # Light grey color (BGR format)

            # Overlay the mask on the camera image
            mask_image[y:y + h, x:x + w] = cv2.addWeighted(mask_image[y:y + h, x:x + w], 0.7, mask_image_roi, 0.3, 0)

    # Draw the template areas on the camera image
    aligned_image = camera_image.copy()
    cv2.drawContours(aligned_image, contours, -1, area_color, 2)

    # Concatenate the aligned image and mask image horizontally
    combined_image = np.concatenate((aligned_image, mask_image), axis=1)

    return combined_image, answer


if __name__ == '__main__':
    # Set the path to the template image
    seg_model = YOLO("yolov8n-seg.pt")

    template_path = "./template3.jpg"
    # Set the threshold for acceptable alignment ratio
    threshold = 0.8
    
    cap = cv2.VideoCapture(1)
    ret, frame = cap.read()
    frame_height, frame_width, _ = frame.shape
    template = cv2.imread(template_path)
    template_resized = cv2.resize(template, (frame_width, frame_height))

    while True:
        ret, frame = cap.read()
        aligned_image, answer  = align_body_part(template_resized, frame, seg_model)
        cv2.imshow("Aligned Image", aligned_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and close windows
    cap.release()
    cv2.destroyAllWindows()