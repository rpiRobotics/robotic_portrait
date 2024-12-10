from RobotRaconteur.Client import *     #import RR client library
import time, traceback, sys, cv2
import numpy as np
from general_robotics_toolbox import * #import general robot toolbox
import pickle
import glob
import threading
from tkinter import messagebox
sys.path.append('toolbox')
from robot_def import *
from lambda_calc import *
from motion_toolbox import *
from portrait import *
sys.path.append('image_processing')
from ClusterImgs import *
sys.path.append('motion_planning')
from PathGenCartesian import *
sys.path.append('robot_motion')
from RobotMotionController import *

ROBOT_NAME='ABB_1200_5_90' # ABB_1200_5_90 or ur5
FORCE_FEEDBACK=True
USE_RR_ROBOT=False

if ROBOT_NAME=='ABB_1200_5_90':
    #########################################################config parameters#########################################################
    robot_cam=robot_obj(ROBOT_NAME,'config/ABB_1200_5_90_robot_default_config.yml',tool_file_path='config/camera.csv')
    robot=robot_obj(ROBOT_NAME,'config/ABB_1200_5_90_robot_default_config.yml',tool_file_path='config/heh6_pen.csv')
    # robot=robot_obj(ROBOT_NAME,'config/ABB_1200_5_90_robot_default_config.yml',tool_file_path='config/brush_pen.csv')
    radius=500 ###eef position to robot base distance w/o z height
    # angle_range=np.array([-3*np.pi/4,-np.pi/4]) ###angle range of joint 1 for robot to move
    angle_range=np.radians([-5,5]) ###angle range of joint 1 for robot to move
    height_range=np.array([750,925]) ###height range for robot to move
    # p_start=np.array([0,-radius,700])	###initial position
    # R_start=np.array([	[0,1,0],
    #                     [0,0,-1],
    #                     [-1,0,0]])	###initial orientation
    p_tracking_start=np.array([ 107.2594, -196.3541,  859.7145])	###initial position
    R_tracking_start=np.array([[ 0.0326 , 0.8737 , 0.4854],
                            [ 0.0888,  0.4812, -0.8721],
                            [-0.9955 , 0.0715, -0.0619]])	###initial orientation
    q_seed=np.zeros(6)
    q_tracking_start=robot_cam.inv(p_tracking_start,R_tracking_start@rot(np.array([0,0,1]),-np.pi/2),q_seed)[0]	###initial joint position
    image_center=np.array([1080,1080])/2	###image center
    RR_robot_sub=RRN.SubscribeService('rr+tcp://localhost:58651?service=robot') if USE_RR_ROBOT else None
    TIMESTEP=0.004
elif ROBOT_NAME=='ur5':
    #########################################################UR config parameters#########################################################
    robot_cam=robot_obj(ROBOT_NAME,'config/ur5_robot_default_config.yml',tool_file_path='config/camera_ur.csv')
    robot=robot_obj(ROBOT_NAME,'config/ur5_robot_default_config.yml',tool_file_path='config/heh6_pen_ur.csv')
    radius=500 ###eef position to robot base distance w/o z height
    angle_range=np.array([-np.pi/4,np.pi/4]) ###angle range of joint 1 for robot to move
    height_range=np.array([500,900]) ###height range for robot to move
    p_tracking_start=np.array([-radius,0,750])	###initial position
    R_tracking_start=np.array([	[0,0,-1],
                        [0,-1,0],
                        [-1,0,0]])	###initial orientation
    q_seed=np.radians([0,-54.8,110,-142,-90,0])
    q_tracking_start=robot.inv(p_tracking_start,R_tracking_start,q_seed)[0]	###initial joint position
    image_center=np.array([1080,1080])/2	###image center
    RR_robot_sub=RRN.SubscribeService('rr+tcp://localhost:58655?service=robot')
    TIMESTEP=0.01
else:
    assert False, "ROBOT_NAME is not valid"

## face track joint 1 adjustment
q_tracking_start[0]=np.mean(angle_range)
T_tracking_start=robot.fwd(q_tracking_start)
p_tracking_start=T_tracking_start.p
R_tracking_start=T_tracking_start.R

RR_ati_cli=None
if FORCE_FEEDBACK:
    RR_ati_cli=RRN.ConnectService('rr+tcp://localhost:59823?service=ati_sensor') # connect to ATI sensor
#########################################################config parameters#########################################################
abb_robot_ip = '192.168.60.101'
paper_size=np.loadtxt('config/paper_size.csv',delimiter=',') # size of the paper
pixel2mm=np.loadtxt('config/pixel2mm.csv',delimiter=',') # pixel to mm ratio
pixel2force=np.loadtxt('config/pixel2force.csv',delimiter=',') # pixel to force ratio
ipad_pose=np.loadtxt('config/ipad_pose.csv',delimiter=',') # ipad pose
H_pentip2ati=np.loadtxt('config/pentip2ati.csv',delimiter=',') # FT sensor info
p_button=np.array([141, -82, 0]) # button position, in ipad frame
R_pencil=Ry(np.pi) # pencil orientation, ipad frame
R_pencil_base=ipad_pose[:3,:3]@R_pencil # pencil orientation, world frame
q_waiting = np.radians([0,-30,25,0,40,0]) # waiting joint position
T_waiting = robot.fwd(q_waiting) # waiting pose, world frame
hover_height=20 # height to hover above the paper
hover_height_close = 1.5
face_track_speed=0.8 # speed to track face
face_track_x = np.array([-np.sin(np.arctan2(p_tracking_start[1],p_tracking_start[0])),np.cos(np.arctan2(p_tracking_start[1],p_tracking_start[0])),0])
face_track_y = np.array([0,0,1])
target_size=[1200,800]
smallest_lam = 20 # smallest path length (unit: mm)
max_stroke_w = 10 # max stroke width
min_stroke_w = 7 # min stroke width
pixelforce_ratio_calib = 1.2 # pixel to force ratio calibration
######## Controller parameters ###
controller_params = {
    "force_ctrl_damping": 60.0, # 200, 180, 90, 60
    "force_epsilon": 0.1, # Unit: N
    "moveL_speed_lin": 10.0, # 10 Unit: mm/sec
    "moveL_acc_lin": 10, # Unit: mm/sec^2 0.6, 1.2, 3.6
    "moveL_speed_ang": np.radians(10), # Unit: rad/sec
    "trapzoid_slope": 1, # trapzoidal load profile. Unit: N/sec
    "load_speed": 15.0, # Unit mm/sec 10
    "unload_speed": 1.0, # Unit mm/sec
    'settling_time': 0.2, # Unit: sec
    "lookahead_time": 0.132, # Unit: sec, 0.02
    "jogging_speed": 100, # Unit: mm/sec
    "jogging_acc": 10, # Unit: mm/sec^2
    'force_filter_alpha': 0.9 # force low pass filter alpha
    }
### Define the motion controller
mctrl=MotionController(robot,ipad_pose,H_pentip2ati,controller_params,TIMESTEP,USE_RR_ROBOT=USE_RR_ROBOT,
                 RR_robot_sub=RR_robot_sub,FORCE_PROTECTION=5,RR_ati_cli=RR_ati_cli,abb_robot_ip=abb_robot_ip)

CART_PLAN = True
JS_PLAN = True

INITIAL_LOGO_PLAN=True

TEMP_DATA_DIR='wen_photo/'

pixel_paths=[]
pixel_paths.append(np.loadtxt(TEMP_DATA_DIR+'face_traj.csv',delimiter=',').reshape((-1,2)))
pixel_paths.append(np.loadtxt(TEMP_DATA_DIR+'hair_traj.csv',delimiter=',').reshape((-1,2)))
pixel_paths.append(np.loadtxt(TEMP_DATA_DIR+'upp_traj.csv',delimiter=',').reshape((-1,2)))
pixel_paths.append(np.loadtxt(TEMP_DATA_DIR+'low_traj.csv',delimiter=',').reshape((-1,2)))
pixel_paths.append(np.loadtxt(TEMP_DATA_DIR+'nose_traj.csv',delimiter=',').reshape((-1,2)))
pixel_paths.append(np.loadtxt(TEMP_DATA_DIR+'l_brow1.csv',delimiter=',').reshape((-1,2)))
pixel_paths.append(np.loadtxt(TEMP_DATA_DIR+'l_brow2.csv',delimiter=',').reshape((-1,2)))
pixel_paths.append(np.loadtxt(TEMP_DATA_DIR+'r_brow1.csv',delimiter=',').reshape((-1,2)))
pixel_paths.append(np.loadtxt(TEMP_DATA_DIR+'r_brow2.csv',delimiter=',').reshape((-1,2)))
pixel_paths.append(np.loadtxt(TEMP_DATA_DIR+'up_r_eye.csv',delimiter=',').reshape((-1,2)))
pixel_paths.append(np.loadtxt(TEMP_DATA_DIR+'up_l_eye.csv',delimiter=',').reshape((-1,2)))
pixel_paths.append(np.loadtxt(TEMP_DATA_DIR+'do_r_eye.csv',delimiter=',').reshape((-1,2)))
pixel_paths.append(np.loadtxt(TEMP_DATA_DIR+'do_l_eye.csv',delimiter=',').reshape((-1,2)))

# create a image with cv2
# draw all the pixels on the image
ratio = 1.2
# real_image_size = [int(343*ratio),int(437*ratio)]
real_image_size = [int(752*ratio),int(1004*ratio)]
image = np.zeros((real_image_size[1],real_image_size[0],3),dtype=np.uint8)
pixel_paths_thickess = []
for pixel_path in pixel_paths:
    pixel_path_resized = pixel_path*ratio
    for pixel in pixel_path_resized:
        image[int(pixel[1]),int(pixel[0]),:]=255
    pixel_paths_thickess.append(np.hstack((pixel_path_resized,np.ones((len(pixel_path_resized),1))*8)))
pixel_paths=pixel_paths_thickess
image_thresh = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# cv2.imshow('img',image_thresh)
# cv2.waitKey(0)

mctrl.start_egm()
q_init = mctrl.read_position()
#########################################################EXECUTION#########################################################
while True:
    start_time=time.time()

    try:
        print("PAGE FLIPPING")
        # mctrl.press_button_routine(p_button,R_pencil,h_offset=hover_height,lin_vel=controller_params['jogging_speed'], q_seed=q_seed)

        img_st = time.time()
        # cv2.imshow('img',img)
        # cv2.waitKey(0)pixel_paths
        ############################################################

        ####################################PLANNING#####################################################
        planning_st = time.time()
        
        print("Image size: ", image_thresh.shape)
        ###Project to IPAD
        print("PROJECTING TO IPAD")
        if CART_PLAN:
            _,cartesian_paths_world,force_paths=image2plane(image_thresh,ipad_pose,pixel2mm,pixel_paths,pixel2force)

        ###Solve Joint Trajectory
        print("SOLVING JOINT TRAJECTORY")
        if JS_PLAN:
            js_paths=[]
            for cartesian_path in cartesian_paths_world:
                curve_js=robot.find_curve_js(cartesian_path,[R_pencil_base]*len(cartesian_path),q_seed)
                js_paths.append(curve_js)
            pickle.dump(js_paths, open(TEMP_DATA_DIR+'js_paths.pkl', 'wb'))
        print("PLANNING TIME: ", time.time()-planning_st)

        ####################################EXECUTION#####################################################
        execution_st = time.time()
        
        print('START DRAWING')
        num_segments = len(js_paths)
        print("NUM SEGMENTS: ", num_segments)
        ###Execute
        try:
            for i in range(0,num_segments):
                if len(js_paths[i])<=1:
                    continue
                cartesian_path_world = cartesian_paths_world[i]
                force_path = force_paths[i]
                curve_xyz = np.dot(mctrl.ipad_pose_inv[:3,:3],cartesian_path_world.T).T+np.tile(mctrl.ipad_pose_inv[:3,-1],(len(cartesian_path_world),1))
                curve_xy = curve_xyz[:,:2] # get xy curve
                # curve_xy = rot([0,0,1],np.pi)@np.array(curve_xy).T
                fz_des = force_path*(-1) # transform to tip desired
                fz_des = fz_des*pixelforce_ratio_calib
                lam = calc_lam_js(js_paths[i],mctrl.robot) # get path length
                # if lam[-1] < smallest_lam:
                #     continue
                traj_q, traj_xy, traj_fz, time_bp = mctrl.trajectory_generate(js_paths[i],curve_xy,fz_des) # get trajectory and time_bp
                #### motion start ###
                # input("Press Enter to start")
                print("Drawing segment ", i)
                mctrl.motion_start_routine(traj_q[0],traj_fz[0],hover_height,hover_height_close,lin_vel=controller_params['jogging_speed'])
                joint_force_exe, cart_force_exe = mctrl.trajectory_force_PIDcontrol(traj_xy,traj_q,traj_fz,force_lookahead=True)
                mctrl.motion_end_routine(traj_q[-1],hover_height, lin_vel=controller_params['jogging_speed'])
        except KeyboardInterrupt:
            print('INTERRUPTED')
            mctrl.motion_end_routine(traj_q[-1],hover_height*4, lin_vel=controller_params['jogging_speed'])

        #jog to end point
        mctrl.motion_end_routine(traj_q[-1],hover_height*4, lin_vel=controller_params['jogging_speed'])

        print('FINISHED DRAWING')

        print("EXECUTION TIME: ", time.time()-execution_st)
        ####################################################################
    except (Exception,KeyboardInterrupt) as e:
        print("Error:", e)
        mctrl.stop_egm()
        break
    
    print('TOTAL TIME: ', time.time()-img_st)
    messagebox.showinfo('Message', 'Next Round?')