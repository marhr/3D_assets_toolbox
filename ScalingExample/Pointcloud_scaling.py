# Mariana Hernandez

import numpy as np
import yaml
import argparse
from scipy.optimize import nnls
import open3d as o3d

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--normalsize',help='Problem definition yaml file ',required=True)
parser.add_argument('--scaledsize',help='Problem definition yaml file ',required=True)
args = parser.parse_args()


with open(args.normalsize, 'r') as f:
    problemdata = yaml.load(f, Loader=yaml.SafeLoader)

A = list(map(float, problemdata['ProblemData']['References']['A'].split(",")))
B = list(map(float, problemdata['ProblemData']['References']['B'].split(",")))
C = list(map(float, problemdata['ProblemData']['References']['C'].split(",")))
D = list(map(float, problemdata['ProblemData']['References']['D'].split(",")))
E = list(map(float, problemdata['ProblemData']['References']['E'].split(","))) # E=heel 
F = list(map(float, problemdata['ProblemData']['References']['F'].split(","))) # F=front of foot

# Scaled-down point coud
with open(args.scaledsize, 'r') as f:
    problemdata_scaled = yaml.load(f, Loader=yaml.SafeLoader)

A_scaled = list(map(float, problemdata_scaled['ProblemData']['References']['A'].split(",")))
B_scaled = list(map(float, problemdata_scaled['ProblemData']['References']['B'].split(",")))
C_scaled = list(map(float, problemdata_scaled['ProblemData']['References']['C'].split(",")))
D_scaled = list(map(float, problemdata_scaled['ProblemData']['References']['D'].split(",")))
E_scaled = list(map(float, problemdata_scaled['ProblemData']['References']['E'].split(",")))
F_scaled = list(map(float, problemdata_scaled['ProblemData']['References']['F'].split(",")))

# x_max = max(A_scaled[0],B_scaled[0],C_scaled[0],D_scaled[0])
# y_max = max(A_scaled[1],B_scaled[1],C_scaled[1],D_scaled[1])
# z_max = max(A_scaled[2],B_scaled[2],C_scaled[2],D_scaled[2])
# x_len = x_max-min(A_scaled[0],B_scaled[0],C_scaled[0],D_scaled[0])
# y_len = y_max-min(A_scaled[1],B_scaled[1],C_scaled[1],D_scaled[1])
# z_len = z_max-min(A_scaled[2],B_scaled[2],C_scaled[2],D_scaled[2])


# print(f"A_scaled\n: {A_scaled}\n")
# print(f"B_scaled\n: {B_scaled}\n")
# print(f"C_scaled\n: {C_scaled}\n")
# print(f"D_scaled\n: {D_scaled}\n")
# print("lens:",x_len,y_len,z_len)

# if z_len<y_len:
	
# 	if z_len<x_len:
# 		 if x_len<y_len:
# 		 	# z smallest --> z becomes y
# 		 	# y becomes z
# 		 	# x remains x
# 		 else:
# 		 	# z becomes x
# 		 	# x becomes z
# 		 	# y remains y
# 	else:
# 		# x is lowest
# 		# x becomes y
# 		# y becomes z
# 		# z becomes x
# else:
# 	if z_len<x_len:
# 		# y becomes z
# 		# x remains x
# 		# z becomes y


CA = np.array(A)-np.array(C)
CB = np.array(B)-np.array(C)
CD = np.array(D)-np.array(C)
EF = np.array(F)-np.array(E)
DA = np.array(A)-np.array(D)

# OA =
# OB = 
# OC = 
print(f"CA,CB,CD,EF, DA:{CA,CB,CD,EF, DA}")

CA_scaled = np.array(A_scaled)-np.array(C_scaled)
CB_scaled = np.array(B_scaled)-np.array(C_scaled)
CD_scaled = np.array(D_scaled)-np.array(C_scaled)
EF_scaled = np.array(F_scaled)-np.array(E_scaled)
DA_scaled = np.array(A_scaled)-np.array(D_scaled)

l1 = np.linalg.norm(CD)
l2 = np.linalg.norm(CB)
l3 = np.linalg.norm(CA)
l4 = np.linalg.norm(EF)
l5 = np.linalg.norm(DA)


a = np.array([[DA_scaled[0]**2, DA_scaled[1]**2, DA_scaled[2]**2], 
	          [CB_scaled[0]**2, CB_scaled[1]**2, CB_scaled[2]**2],
	          [EF_scaled[0]**2, EF_scaled[1]**2, EF_scaled[2]**2]])
b = np.array([l5**2, l2**2, l4**2])
print("Condition number:", np.linalg.cond(a)) # <100 is a good condition number
print(f"A,b,det(A), eig(A):{a}\n{b}\n{np.linalg.det(a)},\n {np.linalg.eig(a)}")

x = np.linalg.solve(a, b)
x_nnls, _ = nnls(a, b)
x_lstsq, *_ = np.linalg.lstsq(a, b, rcond=None)
print(f"x,x_nnls,x_lstsq:{x_nnls,x_lstsq}")
# print(f"x:{x}")
# Adjust generated point cloud dimensions to match the original scale

# Take square root to get the scale factors
scale_factors = np.sqrt(np.abs(x_lstsq))  # Ensure positive even if x has small negatives
# print(f"factors:{scale_factors}")

pcd = o3d.io.read_point_cloud(problemdata_scaled['ProblemData']['DataDir']+problemdata_scaled['ProblemData']['ScanFile'])
pointcloud_scaled = np.array(pcd.points)
# pointcloud_scaled[:,[0,2,1]] = pointcloud_scaled[:,[0,1,2]]



print(f"Points:{pointcloud_scaled}")  
print(f"Diag:{np.diag(np.abs(x))}")
pointcloud_scaled_up = pointcloud_scaled.dot(np.diag(np.abs(scale_factors)))
# x = [scale_factors[0],scale_factors[2],scale_factors[1]]
print(f"Scaled points: {pointcloud_scaled_up}")
print(f"A: {A} \nA_scaled: {np.array(A_scaled).dot(np.diag(np.abs(x)))}\n")
print(f"B: {B} \nB_scaled: {np.array(B_scaled).dot(np.diag(np.abs(x)))}\n")
print(f"C: {C} \nC_scaled: {np.array(C_scaled).dot(np.diag(np.abs(x)))}\n")
print(f"D: {D} \nD_scaled: {np.array(D_scaled).dot(np.diag(np.abs(x)))}\n")
# print(f"A_scaled\n: {A_scaled.dot(np.diag(np.abs(x)))}\n")
# print(f"B_scaled\n: {B_scaled.dot(np.diag(np.abs(x)))}\n")
# print(f"C_scaled\n: {C_scaled.dot(np.diag(np.abs(x)))}\n")
# print(f"D_scaled\n: {D_scaled.dot(np.diag(np.abs(x)))}\n")


# Save or use the rescaled point cloud
rescaled = o3d.geometry.PointCloud()
rescaled.points = o3d.utility.Vector3dVector(pointcloud_scaled_up)
rescaled_ply_name=problemdata_scaled['ProblemData']['DataDir']+"pcd_rescaled_up.ply"
o3d.io.write_point_cloud(rescaled_ply_name, rescaled)


