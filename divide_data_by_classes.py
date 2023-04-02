import os
import shutil
import pandas as pd
# Define the paths to the CSV file and the image folder
csv_path = './Dataset/train.csv'
# img_folder = 'path/to/image/folder'
data_path="./Dataset/"
img_folder=data_path
# Define the output folders for each class
output_folders = ['Retinal_Images/HEALTHY', 'Retinal_Images/MILD', 'Retinal_Images/MODERATE', 'Retinal_Images/SEVERE', 'Retinal_Images/PROLIFERATIVE DR']

def Data_2_classes(data_path,csv_path,output_folders,set_type):

    output_folders= list(map(lambda x: x.replace(x,str(set_type +"_")+x), output_folders))
    img_folder=data_path+set_type+"_images/"

    # Create the output folders if they don't exist
    for folder in output_folders:
        if not os.path.exists(folder):
            os.makedirs(folder)

    class_info = pd.read_csv(csv_path)
    # # id_code,diagnosis
    # # Loop through the rows of the class information
    for index, row in class_info.iterrows():
## Get the filename and class label of the image
        filename = row['id_code']
        class_label = row['diagnosis']
## Define the source and destination paths for the image
        src_path=img_folder+filename+".png"
        new_name="/"+str(class_label)+filename+".png"
        dst_path=output_folders[class_label]+new_name
## Copy the image to the appropriate output folder
        shutil.copy(src_path, dst_path)
    return output_folders

print(Data_2_classes(img_folder,csv_path,output_folders,"train"))
# print(Data_2_classes(img_folder,csv_path,output_folders,"test"))

