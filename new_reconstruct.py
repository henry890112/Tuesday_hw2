import open3d as o3d
import os
import numpy as np
import time
import copy
import cv2
from math import tan


def draw_registration_result(source, target, transformation):
    source.estimate_normals()
    target.estimate_normals()
    source.transform(transformation)
    return source   #轉換後的source

def preprocess_point_cloud(pcd, voxel_size):
    pcd_down = pcd.voxel_down_sample(voxel_size)
    radius_normal = voxel_size * 2
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
    radius_feature = voxel_size * 5
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh


def prepare_dataset(voxel_size,target):
    source = o3d.io.read_point_cloud("./pcd"+str(input_floor)+"/"+str(i+1)+".pcd")
    
    trans_init = np.asarray([[0.0, 0.0, 1.0, 0.0],
                            [1.0, 0.0, 0.0, 0.0], 
                            [0.0, 1.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 1.0]])

    source.transform(trans_init)
    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)
    return source, target, source_down, target_down, source_fpfh, target_fpfh


voxel_size = 0.05  # means 5cm for this dataset

def execute_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size):
    distance_threshold = voxel_size * 1.5
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True, distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
    return result

def refine_registration(source, target, source_fpfh, target_fpfh, voxel_size, result_ransac):
    distance_threshold = voxel_size * 0.4
    source.estimate_normals()
    target.estimate_normals()
   
    result = o3d.pipelines.registration.registration_icp(
        source, target, distance_threshold, result_ransac.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    return result

def estimated_camera_trajectory(transformation, colors, camera_points):
    colors.append([1, 0, 0])  #set the color of red
    translation = np.array([0, 0, 0, 1])
    translation = transformation @ translation   
    translation = translation[0:3].tolist() 
    camera_points.append(translation)
    lines.append([i-1, i])

def depth_image_to_point_cloud(photo_of_number):
    for number in range(1,photo_of_number):
        pointsRGB = []
        pointsXYZ = []
###########此的0000要注意############前面有設錯
        photo_of_number_change = '{0:05d}'.format(number)

        front_rgb = "./ckpt/apartment0/result_predict/apartment"+str(photo_of_number_change)+".png"
        depth_img = "./my_data/depth"+str(input_floor)+"/apartment"+str(photo_of_number_change)+".png"

        img = cv2.imread(front_rgb, 1)   #印出彩色圖片，後面為0則為黑白
        D_img = cv2.imread(depth_img)  #depth_img雖然也是(512,512,3)但3的數字接相同 只是表示的問題而已

        fov_theta = 90
        f = (512/2)*(1/tan(np.radians(fov_theta/2)))

    
        for i in range(512):
            for j in range(512):
                if((i-256)*D_img[i][j][0]/f/25.5 > -0.5):
                    pointsRGB.append([img[i][j][2]/255, img[i][j][1]/255, img[i][j][0]/255])  #open3d use 0~1 for rgb input
                    pointsXYZ.append([(j-256)*D_img[i][j][0]/f/25.5, (i-256)*D_img[i][j][0]/f/25.5, D_img[i][j][0]/25.5])

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pointsXYZ)
        pcd.colors = o3d.utility.Vector3dVector(pointsRGB)

        print("Epoch of create pcd = ", number)
        if not (os.path.exists("./pcd"+str(input_floor))):
            os.makedirs("pcd"+str(input_floor))
        o3d.io.write_point_cloud("./pcd"+str(input_floor)+"/"+str(number)+".pcd", pcd)
        number = number + 1

def true_camera_trajectory():
    true_position = []   #從txt讀取到的資料
    ground_truth_position = [[0,0,0]]
    ground_truth_color = [[0, 0, 0]]
    if int(input_floor) == 0:
        with open('Current_position0.txt', 'r') as fp:           
            for line in fp.readlines():
                line = line.split(" ")
                line = line[0:3]
                true_position.append(line)
    else:
        with open('Current_position1.txt', 'r') as fp:           
            for line in fp.readlines():
                line = line.split(" ")
                line = line[0:3]
                true_position.append(line)

    for i in range(int(number_of_photos)):
        for j in range(3):
            true_position[i][j] = float(true_position[i][j])  #txt讀出來的檔案是str要轉換

        base = [true_position[0][0], true_position[0][1], true_position[0][2]]
        ground_truth_current =  [true_position[i][k] - base[k] for k in range(3)] 
        ground_truth_current[2] = -ground_truth_current[2]

        ground_truth_position.append(ground_truth_current)
        ground_truth_color.append([0,0,0])

    return ground_truth_position, ground_truth_color

if __name__ == '__main__':

    input_floor = input("Input the floor:")   # set the floor
    estimated_camera_points = [[0,0,0]]
    colors = []
    lines = []
    A = []  #whole environment pcd

    fp = open("number"+str(input_floor)+".txt", "r")
    number_of_photos = fp.read()
    fp.close()

    depth_image_to_point_cloud(int(number_of_photos)+1)
    x = o3d.io.read_point_cloud("./pcd"+str(input_floor)+"/1.pcd")
    A.append(x) 
    for i in range(1,int(number_of_photos)):
        source, target, source_down, target_down, source_fpfh, target_fpfh = prepare_dataset(voxel_size,x)
        result_ransac = execute_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size)
        result_icp = refine_registration(source, target, source_fpfh, target_fpfh, voxel_size, result_ransac) #result_icp為經過refine return的值
        y = draw_registration_result(source, target, result_icp.transformation)
        A.append(y)
        x = y
        estimated_camera_trajectory(result_icp.transformation, colors, estimated_camera_points)
        print("Epoch of create the whole environment = ", i)
        i = i + 1

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(estimated_camera_points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)

    # 不印出ground truth有些問題
    # ground_truth_position, ground_truth_color = true_camera_trajectory()
    # line_ground_truth = o3d.geometry.LineSet()
    # line_ground_truth.points = o3d.utility.Vector3dVector(ground_truth_position)
    # line_ground_truth.lines = o3d.utility.Vector2iVector(lines)
    # line_ground_truth.colors = o3d.utility.Vector3dVector(ground_truth_color)

    o3d.visualization.draw_geometries(A+[line_set],
        zoom=0.4459,
        front=[0.9288, -0.2951, -0.2242],
        lookat=[1.6784, 2.0612, 1.4451],
        up=[-0.3402, -0.9189, -0.1996])

    o3d.io.write_point_cloud("./Whole_room.pcd", A[0])
    # o3d.io.write_point_cloud("./trajectory_environment.pcd", A+[line_set])

    print("Done for saving environment and trajectory")

