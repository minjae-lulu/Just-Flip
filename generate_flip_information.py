from PIL import Image
import os
import shutil
import numpy as np
import math
import torch
import json
import re



dir_path = 'data/blender/ficus_test'
new_dir_path = dir_path + "_flip"

shutil.copytree(dir_path, new_dir_path)

new_dir_path = new_dir_path + '/train'

for filename in os.listdir(new_dir_path):
    print("filename: ", filename, "flipped")
    if filename.endswith('.png'):
        img = Image.open(os.path.join(new_dir_path, filename))
        img_flipped = img.transpose(Image.FLIP_LEFT_RIGHT)
        base, ext = os.path.splitext(filename)
        new_filename = f"{base}_flip{ext}"
        img_flipped.save(os.path.join(new_dir_path, new_filename))

print(f"All images have been flipped and saved.")






def Find_Optimal_Sphere(X_cordi,Y_cordi,Z_cordi):
    A = np.zeros((len(X_cordi),4))
    A[:,0] = X_cordi*2
    A[:,1] = Y_cordi*2
    A[:,2] = Z_cordi*2
    A[:,3] = 1
    
    f = torch.zeros((len(X_cordi),1))
    f[:,0] = (X_cordi*X_cordi) + (Y_cordi*Y_cordi) + (Z_cordi*Z_cordi)
    C, residules, rank, singval = np.linalg.lstsq(A,f)
    
    radius2 = (C[0]*C[0])+(C[1]*C[1])+(C[2]*C[2])+C[3]
    radius = math.sqrt(radius2)

    return radius, C[0], C[1], C[2]


def Symmetrically_Transforming(X_cordi,Y_cordi,Z_cordi, A,B,C,D):
    X_flipped_cordi = []
    Y_flipped_cordi = []
    Z_flipped_cordi = []
    
    for i in range(len(X_cordi)):
        X_flipped_cordi.append(X_cordi[i] + 2*A*(-(A*X_cordi[i] + B*Y_cordi[i] + C*Z_cordi[i] + D)/(A*A+B*B+C*C)))
        Y_flipped_cordi.append(Y_cordi[i] + 2*B*(-(A*X_cordi[i] + B*Y_cordi[i] + C*Z_cordi[i] + D)/(A*A+B*B+C*C)))
        Z_flipped_cordi.append(Z_cordi[i] + 2*C*(-(A*X_cordi[i] + B*Y_cordi[i] + C*Z_cordi[i] + D)/(A*A+B*B+C*C)))
    
    X_flipped_cordi = torch.FloatTensor(X_flipped_cordi)
    Y_flipped_cordi = torch.FloatTensor(Y_flipped_cordi)
    Z_flipped_cordi = torch.FloatTensor(Z_flipped_cordi)

    return X_flipped_cordi, Y_flipped_cordi, Z_flipped_cordi




def Projection2Sphere(x,y,z, x0,y0,z0,r):
    a = x*x + y*y + z*z
    b = x*x0 + y*y0 + z*z0 
    c = x0*x0 + y0*y0 + z0*z0 -r*r
    
    alpha1 = (-b + math.sqrt(b*b - 4*a*c) )/(2*a)
    alpha2 = (-b - math.sqrt(b*b - 4*a*c) )/(2*a)
    
    if abs(alpha1-1) > abs(alpha2-1):
        alpha = alpha2
    else:
        alpha = alpha1
    
    return alpha*x, alpha*y, alpha*z
    
    
    
up = torch.FloatTensor([0,0,1])
def Get_Transform_Matrics(x,y,z, x0,y0,z0, up):
    c = torch.stack([x,y,z])
    c = c.squeeze()
    at = torch.FloatTensor([x0,y0,z0])
    at = at.squeeze()
    
    torch0 = torch.FloatTensor([0.])
    temp = torch.FloatTensor([0,0,0,1])

    z_axis = (c-at) / torch.norm(c-at)
    x_axis = torch.cross(up, z_axis) / torch.norm(torch.cross(up, z_axis))
    y_axis = torch.cross(z_axis,x_axis)
    
    z = torch.cat([z_axis, torch0])
    x = torch.cat([x_axis, torch0])
    y = torch.cat([y_axis, torch0])
    
    rotation_mat = torch.stack([x, y, z, temp], dim=-1)
    translation_mat = torch.eye(4)
    translation_mat[:3,3] = c 
    transform_mat = torch.matmul(translation_mat, rotation_mat) 
      
    return transform_mat


up = torch.FloatTensor([0,0,1])
def Find_Optimal_CameraPose (X_cordi, Y_cordi, Z_cordi):
    flipped_camera_pose_set = []
    
    r, x0, y0, z0 = Find_Optimal_Sphere(X_cordi,Y_cordi,Z_cordi)
    X_flipped_cordi, Y_flipped_cordi, Z_flipped_cordi = Symmetrically_Transforming(X_cordi,Y_cordi,Z_cordi, 1,0,0,0)
    
    for i in range(len(X_cordi)):
        X_optim, Y_optim, Z_optim = Projection2Sphere(X_flipped_cordi[i], Y_flipped_cordi[i], Z_flipped_cordi[i], x0,y0,z0,r)
        transform_mat = Get_Transform_Matrics(X_optim.float(), Y_optim.float(), Z_optim.float(), x0,y0,z0, up)
        flipped_camera_pose_set.append(transform_mat)
        
    return flipped_camera_pose_set




def make_flipped_json(file_path, output_path):
    with open(file_path, "r") as file:
        data = json.load(file)
        
    new_frames = []

    X_cordi = [frame["transform_matrix"][0][3] for frame in data["frames"]]
    Y_cordi = [frame["transform_matrix"][1][3] for frame in data["frames"]]
    Z_cordi = [frame["transform_matrix"][2][3] for frame in data["frames"]]

    result = Find_Optimal_CameraPose(torch.FloatTensor(X_cordi), torch.FloatTensor(Y_cordi), torch.FloatTensor(Z_cordi))
    result = [item.tolist() for item in result]

    for i, frame in enumerate(data["frames"]):
        new_frame = frame.copy()
        new_frame["file_path"] = new_frame["file_path"] + "_flip" 
        new_frame["transform_matrix"] = result[i]
        new_frames.append(new_frame)

    data["frames"].extend(new_frames)
    data["frames"].sort(key=lambda frame: (int(re.search(r'\d+', frame["file_path"]).group()), "_flip" in frame["file_path"]))

    with open(output_path, "w") as file:
        json.dump(data, file, indent=4)



base_path = dir_path + '/'
new_dir_path_j = new_dir_path + '/'
input_file_path = os.path.join(base_path, "transforms_train.json")
output_file_path = os.path.join(new_dir_path_j, "flipped_transforms_train.json")

make_flipped_json(input_file_path, output_file_path)
os.rename(output_file_path, input_file_path)