# Mariana H

import argparse
import open3d as o3d
import numpy as np


def read_inp_nodes(file_path):
	
	with open(file_path, "r") as f:
	    lines = f.readlines()

	points = []
	reading_nodes = False
	for line in lines:
		line = line.strip()
		if line.startswith("*NODE"):
		    reading_nodes = True
		    continue
		if reading_nodes:
		    if line.startswith("*"):
		        break
		    parts = line.split(",")
		    if len(parts) >= 4:
		        x, y, z = map(float, parts[1:4])
		        points.append([x, y, z])
	return np.array(points)

# Driver code
parser = argparse.ArgumentParser(description='Convert inp file to ply.')
parser.add_argument('--filename',help='.inp file address',required=True)
args = parser.parse_args()

name = args.filename.split("/")[1].split(".")[0]
print(name)

# --- STEP 2: Create point cloud and save as .ply ---
points = read_inp_nodes(args.filename)

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)

# Optional: Estimate normals (recommended before surface reconstruction later)
pcd.estimate_normals()

# Save as PLY
o3d.io.write_point_cloud(name+".ply", pcd)
print("Saved point cloud as "+name+".ply")