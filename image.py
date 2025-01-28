# # STEP 1
# # import libraries
# import fitz  # PyMuPDF
# import io
# from PIL import Image

# # STEP 2
# # file path you want to extract images from
# file = "C:\\Users\\Heeta Parmar\\OneDrive - Galaxy Office Automation Pvt Ltd\\Desktop\\data\\manuals\\microsoft_surface_3.pdf"

# # open the file
# pdf_file = fitz.open(file)

# # STEP 3
# # iterate over PDF pages
# for page_index in range(len(pdf_file)):

#     # get the page itself
#     page = pdf_file.load_page(page_index)  # load the page
#     image_list = page.get_images(full=True)  # get images on the page

#     # printing number of images found in this page
#     if image_list:
#         print(f"[+] Found a total of {len(image_list)} images on page {page_index}")
#     else:
#         print("[!] No images found on page", page_index)
    
#     for image_index, img in enumerate(image_list, start=1):
#         # get the XREF of the image
#         xref = img[0]

#         # extract the image bytes
#         base_image = pdf_file.extract_image(xref)
#         image_bytes = base_image["image"]

#         # get the image extension
#         image_ext = base_image["ext"]

#         # save the image
#         image_name = f"image{page_index+1}_{image_index}.{image_ext}"
#         with open(image_name, "wb") as image_file:
#             image_file.write(image_bytes)
#             print(f"[+] Image saved as {image_name}")


# import fitz  # PyMuPDF
# import os

# def extract_images_with_context(pdf_path, output_dir="extracted_images", page_range=None):
#     """
#     Extracts images from a specific page range of a PDF and saves them with their context.

#     Args:
#         pdf_path (str): Path to the input PDF file.
#         output_dir (str): Directory to save extracted images.
#         page_range (tuple): A tuple specifying the start and end pages (1-based indexing).
    
#     Returns:
#         list: A list of dictionaries with image file paths and their associated context.
#     """
#     # Ensure the output directory exists
#     os.makedirs(output_dir, exist_ok=True)

#     # Open the PDF document
#     doc = fitz.open(pdf_path)
#     results = []

#     # Default to all pages if no range is provided
#     start_page, end_page = (0, len(doc)) if page_range is None else (page_range[0] - 1, page_range[1])

#     for page_number in range(start_page, end_page):
#         page = doc[page_number]
#         images = page.get_images(full=True)  # Get all images on the page

#         # Skip pages without images
#         if not images:
#             print(f"No images found on page {page_number + 1}")
#             continue

#         # Extract each image
#         for i, img in enumerate(images):
#             try:
#                 xref = img[0]  # XREF of the image
#                 base_image = doc.extract_image(xref)
#                 image_bytes = base_image["image"]
#                 image_ext = base_image["ext"]  # Image extension (e.g., jpg, png)

#                 # Save the image to the output directory
#                 image_filename = f"page_{page_number+1}_image_{i+1}.{image_ext}"
#                 image_path = os.path.join(output_dir, image_filename)
#                 with open(image_path, "wb") as image_file:
#                     image_file.write(image_bytes)

#                 # Get the bounding box of the image
#                 bbox = img[2]
#                 if isinstance(bbox, (tuple, list)) and len(bbox) == 4:
#                     bbox_rect = fitz.Rect(bbox)  # Valid bounding box
#                 else:
#                     print(f"Invalid bounding box for image {i+1} on page {page_number+1}: {bbox}. Using default fallback.")
#                     bbox_rect = fitz.Rect(0, 0, page.rect.width, page.rect.height)  # Default to full page

#                 # Get surrounding text
#                 surrounding_text = page.get_text("text", clip=bbox_rect)

#                 # Store results
#                 results.append({
#                     "image_path": image_path,
#                     "context": surrounding_text.strip(),
#                     "page": page_number + 1
#                 })

#             except Exception as e:
#                 print(f"Error processing image {i+1} on page {page_number+1}: {e}")
#                 continue

#     # Close the PDF document
#     doc.close()

#     return results


# # PDF path provided
# pdf_path = r"C:\Users\Heeta Parmar\OneDrive - Galaxy Office Automation Pvt Ltd\Desktop\data\manuals\microsoft_surface_3.pdf"

# # Directory where images will be saved
# output_directory = "output_images"

# # Specify a page range (optional, 1-based indexing)
# page_range = (1,20)  # Pages 1–10; adjust as needed or set to None to process all pages

# # Extract images and their contexts
# extracted_data = extract_images_with_context(pdf_path, output_directory, page_range)

# # Print the results
# for data in extracted_data:
#     print(f"Image Path: {data['image_path']}")
#     print(f"Context: {data['context']}")
#     print(f"Page: {data['page']}")
#     print("-" * 50)


import pathlib
from pathlib import Path
from pdf2image import convert_from_path
#converts each pdf page into images and then scannes for text in the images
def pdf_to_image(pdf_path, output_folder: str = "."):
   """
   A function to convert PDF files to images
   """
   # Create the output folder if it doesn't exist
   if not Path(output_folder).exists():
       Path(output_folder).mkdir()

   pages = convert_from_path(pdf_path, output_folder=output_folder, fmt="png")

   return pages
pdf_path ="C:\\Users\\Heeta Parmar\\OneDrive - Galaxy Office Automation Pvt Ltd\\Desktop\\data\\manuals\\dell_latitutde_7350.pdf"

pdf_to_image(pdf_path, output_folder="documents")
