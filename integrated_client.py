from RobotRaconteur.Client import *     #import RR client library
import time, traceback, sys, cv2
import numpy as np
sys.path.append('toolbox')
from robot_def import *
from vel_emulate_sub import EmulatedVelocityControl
from lambda_calc import *
from motion_toolbox import *
from portrait import *
from traversal_force import *
from traj_gen_cartesian import *
from execute import *


#########################################################config parameters#########################################################
robot_cam=robot_obj('ABB_1200_5_90','config/ABB_1200_5_90_robot_default_config.yml',tool_file_path='config/camera.csv')
robot=robot_obj('ABB_1200_5_90','config/ABB_1200_5_90_robot_default_config.yml',tool_file_path='config/heh6_pen2.csv')
radius=500 ###eef position to robot base distance w/o z height
angle_range=np.array([-3*np.pi/4,-np.pi/4]) ###angle range for robot to move
height_range=np.array([500,900]) ###height range for robot to move
p_start=np.array([0,-radius,700])	###initial position
R_start=np.array([	[0,1,0],
					[0,0,-1],
					[-1,0,0]])	###initial orientation
q_start=robot_cam.inv(p_start,R_start,np.zeros(6))	###initial joint position
image_center=np.array([1080,1080])/2	###image center

#########################################################RR PARAMETERS#########################################################
RR_robot_sub=RRN.SubscribeService('rr+tcp://localhost:58651?service=robot')
RR_robot=RR_robot_sub.GetDefaultClientWait(1)
robot_state = RR_robot_sub.SubscribeWire("robot_state")
robot_const = RRN.GetConstants("com.robotraconteur.robotics.robot", RR_robot)
halt_mode = robot_const["RobotCommandMode"]["halt"]
position_mode = robot_const["RobotCommandMode"]["position_command"]
trajectory_mode = robot_const["RobotCommandMode"]["trajectory"]
jog_mode = robot_const["RobotCommandMode"]["jog"]
RobotJointCommand = RRN.GetStructureType("com.robotraconteur.robotics.robot.RobotJointCommand",RR_robot)
RR_robot.command_mode = halt_mode
time.sleep(0.1)
# RR_robot.reset_errors()
# time.sleep(0.1)

RR_robot.command_mode = position_mode
cmd_w = RR_robot_sub.SubscribeWire("position_command")

vel_ctrl = EmulatedVelocityControl(RR_robot,robot_state, cmd_w)
#enable velocity mode
vel_ctrl.enable_velocity_mode()

########subscription mode
def connect_failed(s, client_id, url, err):
	print ("Client connect failed: " + str(client_id.NodeID) + " url: " + str(url) + " error: " + str(err))

face_tracking_sub=RRN.SubscribeService('rr+tcp://localhost:52222/?service=Face_tracking')
obj = face_tracking_sub.GetDefaultClientWait(1)		#connect, timeout=30s
bbox_wire=face_tracking_sub.SubscribeWire("bbox")
image_wire=face_tracking_sub.SubscribeWire("frame_stream")

face_tracking_sub.ClientConnectFailed += connect_failed

start_time=time.time()
#jog to initial_position
jog_joint(q_start,vel_ctrl)
while True:
	wire_packet=bbox_wire.TryGetInValue()
	if wire_packet[0]:
		bbox=wire_packet[1]
		print(len(bbox))
		if len(bbox)==0: #if no face detected, jog to initial position
			diff=q_start-vel_ctrl.joint_position()
			if np.linalg.norm(diff)>0.1:
				qdot=diff/np.linalg.norm(diff)
			else:
				qdot=diff
			
			vel_ctrl.set_velocity_command(qdot)
			
		else:	#if face detected
			q_cur=vel_ctrl.joint_position()
			pose_cur=robot_cam.fwd(q_cur)
			if q_cur[0]<angle_range[0] or q_cur[0]>angle_range[1] or pose_cur.p[2]<height_range[0] or pose_cur.p[2]>height_range[1]:
				vel_ctrl.set_velocity_command(np.zeros(6))
				continue
			#calculate size of bbox
			size=np.array([bbox[2]-bbox[0],bbox[3]-bbox[1]])
			#calculate center of bbox
			center=np.array([bbox[0]+size[0]/2,bbox[1]+size[1]/2])
			z_gain=-1
			x_gain=-1e-3
			zd=center[1]-image_center[1]
			xd=center[0]-image_center[0]
			
			try:
				q_temp=robot_cam.inv(pose_cur.p+zd*np.array([0,0,z_gain]),pose_cur.R,q_cur)
			except:
				continue
			q_temp+=xd*np.array([x_gain,0,0,0,0,0])
			q_diff=q_temp-q_cur
			if np.linalg.norm(q_diff)>0.5:
				qdot=0.5*q_diff/np.linalg.norm(q_diff)
			else:
				qdot=q_diff
			vel_ctrl.set_velocity_command(qdot)
			
			if np.linalg.norm(qdot)<0.01:
				if time.time()-start_time>5:
					break
			else:
				start_time=time.time()

vel_ctrl.disable_velocity_mode()

RR_image=image_wire.TryGetInValue()
if RR_image[0]:
	img=RR_image[1]
	img=np.array(img.data,dtype=np.uint8).reshape((img.image_info.height,img.image_info.width,3))

print('IMAGE TAKEN')
cv2.imshow("img", img)
cv2.waitKey(0)

###############################################################################PLANNING########################################################################################
paper_size=np.loadtxt('config/paper_size.csv',delimiter=',')
pen_radius=np.loadtxt('config/pen_radius.csv',delimiter=',')
ipad_pose=np.loadtxt('config/ipad_pose.csv',delimiter=',')

###portrait GAN
anime = AnimeGANv3('models/AnimeGANv3_PortraitSketch.onnx')
output = anime.forward(img)
img_gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
pixel2mm=min(paper_size/img_gray.shape)

cv2.imshow("img", output)
cv2.waitKey(0)
###Pixel Traversal

pixel_paths=image_traversal(img,paper_size,pen_radius)
###Project to IPAD
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

cartesian_paths=image2plane(img, ipad_pose, pixel2mm,pixel_paths)

for cartesian_path in cartesian_paths:
	###plot out the path in 3D
	ax.plot(cartesian_path[:,0], cartesian_path[:,1], cartesian_path[:,2], 'b')
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
plt.show()

###Solve Joint Trajectory
R_pencil=np.array([ [-1,0,0],
					[0,1,0],
					[0,0,-1]])
js_paths=[]
for cartesian_path in cartesian_paths:
	curve_js=robot.find_curve_js(cartesian_path,[R_pencil]*len(cartesian_path),np.zeros(6))
	js_paths.append(curve_js)

###Execute
start=True
for i in range(len(js_paths)):
	cartesian_path=cartesian_paths[i]
	curve_js=js_paths[i]
	if len(curve_js)>1:
		pose_start=robot.fwd(curve_js[0])
		if start:
			#jog to starting point
			p_start=pose_start.p+20*ipad_pose[:3,-2]
			q_start=robot.inv(p_start,pose_start.R,curve_js[0])[0]
			jog_joint_position_cmd(q_start)
			jog_joint_position_cmd(curve_js[0],accurate=True)
			start=False
		else:
			pose_cur=robot.fwd(robot_state.InValue.joint_position)
			p_mid=(pose_start.p+pose_cur.p)/2+10*ipad_pose[:3,-2]
			q_mid=robot.inv(p_mid,pose_start.R,curve_js[0])[0]
			#arc-like trajectory to next segment
			trajectory_position_cmd(np.vstack((robot_state.InValue.joint_position,q_mid,curve_js[0])),v=0.1)
			jog_joint_position_cmd(curve_js[0])

		#drawing trajectory
		trajectory_position_cmd(curve_js,v=0.1)
		#jog to end point in case
		jog_joint_position_cmd(curve_js[-1])

#jog to end point
pose_end=robot.fwd(curve_js[-1])
p_end=pose_end.p+20*ipad_pose[:3,-2]
q_end=robot.inv(p_end,pose_end.R,curve_js[-1])[0]
jog_joint_position_cmd(q_end)