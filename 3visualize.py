import os
import cv2
import pydicom

def data_visual(dicom_path, png_path):
   
    dcm = pydicom.dcmread(dicom_path)
    img = dcm.pixel_array
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
    cv2.imwrite(png_path, img)

def convert_dataset(dicom_dir, out_dir):
   
    for root, dirs, files in os.walk(dicom_dir):
        for file in files:
            if file.endswith(".dcm"):
              
                dicom_path = os.path.join(root, file)

                
                relative_path = os.path.relpath(root, dicom_dir)
                save_dir = os.path.join(out_dir, relative_path)
                os.makedirs(save_dir, exist_ok=True)

                png_name = file.replace(".dcm", ".png")
                png_path = os.path.join(save_dir, png_name)

                data_visual(dicom_path, png_path)
                print(f"Converted: {dicom_path} -> {png_path}")

if __name__ == "__main__":
 
    dicom_dir = r"C:\Users\Sakshi\OneDrive\Desktop\clinico_App\chest_xray"
    out_dir = r"C:\Users\Sakshi\OneDrive\Desktop\clinico_App\png_data"

    convert_dataset(dicom_dir, out_dir)
    print("✅ All DICOM files converted to PNGs!")
