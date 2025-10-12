import os
import cv2
import pydicom

def data_pre(dicom_path, png_path):
    dcm = pydicom.dcmread(dicom_path)
    img = dcm.pixel_array
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
    cv2.imwrite(png_path, img)

def convert_dataset(dicom_dir, out_dir):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    for root, dirs, files in os.walk(dicom_dir):
        for file in files:
            if file.endswith(".dcm"):
                dicom_path = os.path.join(root, file)
                png_name = file.replace(".dcm", ".png")
                png_path = os.path.join(out_dir, png_name)
                data_pre(dicom_path, png_path)
                print(f"Converted: {dicom_path} -> {png_path}")

# Example usage (adjust these paths)
dicom_dir = r"C:\Users\Sakshi\OneDrive\Desktop\clinico_App\chest_xray"
out_dir = r"C:\Users\Sakshi\OneDrive\Desktop\clinico_App\png_data"

convert_dataset(dicom_dir, out_dir)
 