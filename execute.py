from RobotRaconteur.Client import *
import numpy as np
import matplotlib.pyplot as plt
import glob, cv2, sys, time
from scipy.interpolate import CubicSpline

sys.path.append('toolbox')
from robot_def import *
from vel_emulate_sub import EmulatedVelocityControl
from lambda_calc import *
from motion_toolbox import *

def jog_joint_position_cmd(q,v=0.5,accurate=False):
	global RobotJointCommand, cmd_w, command_seqno, robot_state

	total_time=np.linalg.norm(q-robot_state.InValue.joint_position)/v

	start_time=time.time()
	while time.time()-start_time<total_time:
		# Increment command_seqno
		command_seqno += 1
		# Create Fill the RobotJointCommand structure
		joint_cmd = RobotJointCommand()
		joint_cmd.seqno = command_seqno # Strictly increasing command_seqno
		joint_cmd.state_seqno = robot_state.InValue.seqno # Send current robot_state.seqno as failsafe
		
		# Set the joint command
		frac=(time.time()-start_time)/total_time
		joint_cmd.command = frac*q+(1-frac)*robot_state.InValue.joint_position
	
		# Send the joint command to the robot
		cmd_w.SetOutValueAll(joint_cmd)
	
	if accurate:
		###additional points for accuracy
		start_time=time.time()
		while time.time()-start_time<1:
			# Increment command_seqno
			command_seqno += 1
			# Create Fill the RobotJointCommand structure
			joint_cmd = RobotJointCommand()
			joint_cmd.seqno = command_seqno # Strictly increasing command_seqno
			joint_cmd.state_seqno = robot_state.InValue.seqno # Send current robot_state.seqno as failsafe
			
			# Set the joint command
			joint_cmd.command = q

			# Send the joint command to the robot
			cmd_w.SetOutValueAll(joint_cmd)


def trajectory_position_cmd(q_all,v=0.5):
	global RobotJointCommand, cmd_w, command_seqno, robot_state


	lamq_bp=[0]
	for i in range(len(q_all)-1):
		lamq_bp.append(lamq_bp[-1]+np.linalg.norm(q_all[i+1]-q_all[i]))
	time_bp=np.array(lamq_bp)/v
	seg=1

	start_time=time.time()
	while time.time()-start_time<time_bp[-1]:
		# now=time.time()
		# Increment command_seqno
		command_seqno += 1
		# Create Fill the RobotJointCommand structure
		joint_cmd = RobotJointCommand()
		joint_cmd.seqno = command_seqno # Strictly increasing command_seqno
		joint_cmd.state_seqno = robot_state.InValue.seqno # Send current robot_state.seqno as failsafe
		

		#find current segment
		if time.time()-start_time>time_bp[seg]:
			seg+=1
		if seg==len(q_all):
			break
		frac=(time.time()-start_time-time_bp[seg-1])/(time_bp[seg]-time_bp[seg-1])

		# Set the joint command
		joint_cmd.command = frac*q_all[seg]+(1-frac)*q_all[seg-1]

		# Send the joint command to the robot
		cmd_w.SetOutValueAll(joint_cmd)

		# print(time.time()-now)


		


def spline_js(cartesian_path,curve_js,vd,rate=250):
	lam=calc_lam_cs(cartesian_path)
	# polyfit=np.polyfit(lam,curve_js,deg=40)
	# lam=np.linspace(0,lam[-1],int(rate*lam[-1]/vd))
	# return np.vstack((np.poly1d(polyfit[:,0])(lam), np.poly1d(polyfit[:,1])(lam), np.poly1d(polyfit[:,2])(lam), np.poly1d(polyfit[:,3])(lam), np.poly1d(polyfit[:,4])(lam), np.poly1d(polyfit[:,5])(lam))).T
	polyfit=CubicSpline(lam,curve_js)
	lam=np.linspace(0,lam[-1],int(rate*lam[-1]/vd))
	return polyfit(lam)

def main():
	global  RobotJointCommand, cmd_w, command_seqno, robot_state
	rate=250
	img_name='wen_out'
	ipad_pose=np.loadtxt('config/ipad_pose.csv',delimiter=',')
	num_segments=len(glob.glob('path/cartesian_path/'+img_name+'/*.csv'))
	robot=robot_obj('ABB_1200_5_90','config/ABB_1200_5_90_robot_default_config.yml',tool_file_path='config/heh6_pen2.csv')
	##RR PARAMETERS
	RR_robot_sub=RRN.SubscribeService('rr+tcp://localhost:58651?service=robot')
	RR_robot=RR_robot_sub.GetDefaultClientWait(1)
	robot_state = RR_robot_sub.SubscribeWire("robot_state")
	robot_const = RRN.GetConstants("com.robotraconteur.robotics.robot", RR_robot)
	halt_mode = robot_const["RobotCommandMode"]["halt"]
	position_mode = robot_const["RobotCommandMode"]["position_command"]
	trajectory_mode = robot_const["RobotCommandMode"]["trajectory"]
	jog_mode = robot_const["RobotCommandMode"]["jog"]
	RobotJointCommand = RRN.GetStructureType("com.robotraconteur.robotics.robot.RobotJointCommand",RR_robot)
	command_seqno = 1
	RR_robot.command_mode = halt_mode
	time.sleep(0.1)
	# RR_robot.reset_errors()
	# time.sleep(0.1)

	RR_robot.command_mode = position_mode
	cmd_w = RR_robot_sub.SubscribeWire("position_command")


	for i in range(num_segments):
		cartesian_path=np.loadtxt('path/cartesian_path/'+img_name+'/%i.csv'%i,delimiter=',').reshape((-1,3))
		curve_js=np.loadtxt('path/js_path/'+img_name+'/%i.csv'%i,delimiter=',').reshape((-1,6))
		print(i)
		if len(curve_js)>1:

			#jog to starting point
			pose_start=robot.fwd(curve_js[0])
			p_start=pose_start.p+20*ipad_pose[:3,-2]
			q_start=robot.inv(p_start,pose_start.R,curve_js[0])[0]
			jog_joint_position_cmd(q_start)
			jog_joint_position_cmd(curve_js[0],accurate=True)
			trajectory_position_cmd(curve_js,v=0.1)
			
			#jog to end point
			pose_end=robot.fwd(curve_js[-1])
			p_end=pose_end.p+20*ipad_pose[:3,-2]
			q_end=robot.inv(p_end,pose_end.R,curve_js[-1])[0]
			jog_joint_position_cmd(q_end)
			time.sleep(1)
			

if __name__ == '__main__':
	main()