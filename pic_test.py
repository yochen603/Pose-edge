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
    answer= False

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
            masks = roi_results[0].masks.data

            # Resize the mask to match the size of the contour
            mask = masks[0].cpu().numpy()
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

            # Calculate the ratio of the masked area to the contour area
            contour_area = cv2.contourArea(contour)
            mask_area = np.sum(mask)
            print("mask_area", mask_area)
            print("contour_area", contour_area)
            mask_ratio = mask_area / contour_area
            print("mask_ratio", mask_ratio)
            # Update the alignment ratio and area color if a better alignment is found
            if mask_ratio > threshold:
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

    return combined_image,answer


if __name__ == '__main__':
    # Set the path to the template image
    seg_model = YOLO("yolov8n-seg.pt")

    template_path = "./template3.jpg"
    # Set the threshold for acceptable alignment ratio
    threshold = 0.2

    # Set the path to the input picture
    input_picture_path = "./test3.jpg"

    # Read the input picture
    input_picture = cv2.imread(input_picture_path)
    print("输入图片的大小",input_picture.shape)
    # Read the template image and resize it to match the input picture size
    template = cv2.imread(template_path)
    template_resized = cv2.resize(template, (input_picture.shape[1], input_picture.shape[0]))
    print("输入模板的大小",template_resized.shape)
    # Align the body part in the input picture
    aligned_image, answer = align_body_part(template_resized, input_picture, seg_model)

    # Display the aligned image
    cv2.imshow("Aligned Image", aligned_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()