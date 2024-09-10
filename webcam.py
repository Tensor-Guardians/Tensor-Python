import cv2
import pytesseract
from pdf2image import convert_from_path
from PIL import Image
import os
import re
import numpy as np
from skimage.metrics import structural_similarity as ssim

reference_image_path = r"C:\Users\91629\OneDrive\Desktop\1.jpg"  
reference_image = cv2.imread(reference_image_path)

#  webcam
def capture_image():
    cap = cv2.VideoCapture(0) 
    ret, frame = cap.read()
    if ret:
        cv2.imshow("Captured Image", frame)
        cv2.imwrite('captured_image.jpg', frame) 
    cap.release()
    cv2.destroyAllWindows()
    return 'captured_image.jpg'  
captured_image_path = capture_image() 


def preprocess_image(image):
    
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    denoised_image = cv2.fastNlMeansDenoising(gray_image, h=30)
    equalized_image = cv2.equalizeHist(denoised_image)
    kernel = np.array([[0, -1, 0], [-1, 5,-1], [0, -1, 0]])  
    sharpened_image = cv2.filter2D(equalized_image, -1, kernel)
    resized_image = cv2.resize(sharpened_image, (500, 500))
    
    return resized_image


reference_image = preprocess_image(reference_image)
captured_image = preprocess_image(cv2.imread(captured_image_path))


def compare_images(img1, img2):
   
    similarity_index, diff = ssim(img1, img2, full=True)
    
   
    diff = (diff * 255).astype("uint8")
    
    return similarity_index, diff


similarity, diff_image = compare_images(reference_image, captured_image)


if similarity > 0.9:  
    print(f"Images match with similarity: {similarity}")
else:
    print(f"Images do not match. Similarity: {similarity}")


def display_images(img1, img2, diff_img):
    
    combined = np.hstack((img1, img2, diff_img))  # Stack images horizontally
    cv2.imshow("Reference | Captured | Difference", combined)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


display_images(reference_image, captured_image, diff_image)



def capture_image():
    
    cap = cv2.VideoCapture(0) 
    ret, frame = cap.read()
    if ret:
        cv2.imshow("Captured Image", frame)
        cv2.imwrite('captured_image.jpg', frame)  
    cap.release()
    cv2.destroyAllWindows()
    return 'captured_image.jpg'  

captured_image_path = capture_image()  


# Image Preprocessing for OCR
def preprocess_image_for_ocr(image_path):
    
    img = cv2.imread(image_path)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    thresh_img = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                       cv2.THRESH_BINARY, 11, 2)
    
 
    processed_image_path = "processed_image.jpg"
    cv2.imwrite(processed_image_path, thresh_img)
    
    return processed_image_path


def ocr_image(image_path):
    
    processed_image_path = preprocess_image_for_ocr(image_path)  
    img = Image.open(processed_image_path)
    
   
    config = "--oem 3 --psm 6"  
    text = pytesseract.image_to_string(img, config=config)
    
    return text



def preprocess_text(text):
    
    text = re.sub(r'\s+', ' ', text).strip().lower()
    return text



reference_image_path = r"C:\Users\91629\OneDrive\Desktop\1.jpg"  
image_text1 = ocr_image(reference_image_path)
processed_image_text1 = preprocess_text(image_text1)


captured_image_text = ocr_image(captured_image_path)
processed_image_text2 = preprocess_text(captured_image_text)


if processed_image_text1 == processed_image_text2:
    print("The texts match exactly.")
else:
    print("The texts do not match.")

print("Reference Image Text: ", processed_image_text1)
print("Captured Image Text: ", processed_image_text2)
