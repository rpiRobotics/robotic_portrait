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
    "force_ctrl_damping": 90.0, # 200, 180, 90, 60
    "force_epsilon": 0.1, # Unit: N
    "moveL_speed_lin": 6.0, # 10 Unit: mm/sec
    "moveL_acc_lin": 7.2, # Unit: mm/sec^2 0.6, 1.2, 3.6
    "moveL_speed_ang": np.radians(10), # Unit: rad/sec
    "trapzoid_slope": 1, # trapzoidal load profile. Unit: N/sec
    "load_speed": 15.0, # Unit mm/sec 10
    "unload_speed": 1.0, # Unit mm/sec
    'settling_time': 0.2, # Unit: sec
    "lookahead_time": 0.132, # Unit: sec, 0.02
    "jogging_speed": 200, # Unit: mm/sec
    "jogging_acc": 60, # Unit: mm/sec^2
    'force_filter_alpha': 0.9 # force low pass filter alpha
    }
### Define the motion controller
mctrl=MotionController(robot,ipad_pose,H_pentip2ati,controller_params,TIMESTEP,USE_RR_ROBOT=USE_RR_ROBOT,
                 RR_robot_sub=RR_robot_sub,FORCE_PROTECTION=5,RR_ati_cli=RR_ati_cli,abb_robot_ip=abb_robot_ip)

###### Face tracking RR client ######
def connect_failed(s, client_id, url, err):
    print ("Client connect failed: " + str(client_id.NodeID) + " url: " + str(url) + " error: " + str(err))
face_tracking_sub=RRN.SubscribeService('rr+tcp://localhost:52222/?service=Face_tracking')
obj = face_tracking_sub.GetDefaultClientWait(1)		#connect, timeout=30s
bbox_wire=face_tracking_sub.SubscribeWire("bbox")
image_wire=face_tracking_sub.SubscribeWire("frame_stream")
face_tracking_sub.ClientConnectFailed += connect_failed

### Portrait NNs ###
faceseg = FaceSegmentation()
anime = AnimeGANv3('models/AnimeGANv3_PortraitSketch.onnx')

# print(robot.fwd(mctrl.read_position()))
# exit()
TEMP_DATA_DIR = 'temp_data/'
# TEMP_DATA_DIR = 'imgs/'
test_logo='logos_words'

TAKE_FACE_IMAGE = True
FACE_PORTRAIT = True
PIXEL_PLAN = True
CART_PLAN = True
JS_PLAN = True

INITIAL_LOGO_PLAN=True

img_names = ['alex','brandon','eric','glenn','julia','molly','thea','wen']
img_count = 0

mctrl.start_egm()
q_init = mctrl.read_position()
#########################################################EXECUTION#########################################################
while True:
    start_time=time.time()
    
    def robot_thinking():
        print('Robot thinking')
        origin_acc = mctrl.params["moveL_acc_lin"] 
        origin_lookaheadt = mctrl.params['lookahead_time']
        origin_fdamp = mctrl.params['force_ctrl_damping']
        mctrl.params['lookahead_time'] = 0.04
        mctrl.params["moveL_acc_lin"] = origin_acc
        mctrl.params['force_ctrl_damping'] = origin_fdamp
        while t_robot_thinking_flag:
            ########### Logo words ###########
            ####### write words
            num_segments=len(glob.glob('path/pixel_path/'+test_logo+'/*.csv'))
            
            if INITIAL_LOGO_PLAN:
                repeat_row = 4
                repeat_col = 3
                img=cv2.imread('imgs/'+test_logo+'_resized.png')
                # replocate the image and pixel paths
                img_new = np.ones((img.shape[0]*repeat_row,img.shape[1]*repeat_col,3))*255
                pixel_paths=[]
                for i in range(repeat_row*repeat_col):
                    pixel_offset = np.array([(i%repeat_col)*img.shape[1],(i//repeat_col)*img.shape[0]])
                    img_new[pixel_offset[1]:img.shape[0]+pixel_offset[1],pixel_offset[0]:pixel_offset[0]+img.shape[1],:]=img
                    for j in range(num_segments):
                        pixel_paths.append(np.loadtxt('path/pixel_path/'+test_logo+'/%i.csv'%j,delimiter=',').reshape((-1,3)))
                        pixel_paths[-1][:,:2]+=pixel_offset
                img = img_new
                if not t_robot_thinking_flag:
                    break
                _,cartesian_paths_world,force_paths=image2plane(img,ipad_pose,pixel2mm,pixel_paths,pixel2force)
                pickle.dump(cartesian_paths_world, open(TEMP_DATA_DIR+'cartesian_paths_world_logo.pkl', 'wb'))
                pickle.dump(force_paths, open(TEMP_DATA_DIR+'force_paths_logo.pkl', 'wb'))
                if not t_robot_thinking_flag:
                    break
                js_paths=[]
                for cartesian_path in cartesian_paths_world:
                    curve_js=robot.find_curve_js(cartesian_path,[R_pencil_base]*len(cartesian_path),q_seed)
                    js_paths.append(curve_js)
                pickle.dump(js_paths, open(TEMP_DATA_DIR+'js_paths_logo.pkl', 'wb'))
            else:
                print("Read from exsiting.")
                cartesian_paths_world = pickle.load(open(TEMP_DATA_DIR+'cartesian_paths_world_logo.pkl', 'rb'))
                force_paths = pickle.load(open(TEMP_DATA_DIR+'force_paths_logo.pkl', 'rb'))
                js_paths = pickle.load(open(TEMP_DATA_DIR+'js_paths_logo.pkl', 'rb'))
            
            if not t_robot_thinking_flag:
                break
            
            num_segments = len(js_paths)
            ###Execute
            try:
                mctrl.press_button_routine(p_button,R_pencil,h_offset=hover_height,lin_vel=controller_params['jogging_speed'], q_seed=q_seed)
                for i in range(0,num_segments):
                    if len(js_paths[i])<=1:
                        continue
                    cartesian_path_world = cartesian_paths_world[i]
                    force_path = force_paths[i]
                    curve_xyz = np.dot(mctrl.ipad_pose_inv[:3,:3],cartesian_path_world.T).T+np.tile(mctrl.ipad_pose_inv[:3,-1],(len(cartesian_path_world),1))
                    curve_xy = curve_xyz[:,:2] # get xy curve
                    fz_des = force_path*(-1) # transform to tip desired
                    # fz_des = fz_des*pixelforce_ratio_calib
                    lam = calc_lam_js(js_paths[i],mctrl.robot) # get path length
                    if lam[-1] < smallest_lam:
                        continue
                    if not t_robot_thinking_flag:
                        break
                    traj_q, traj_xy, traj_fz, time_bp = mctrl.trajectory_generate(js_paths[i],curve_xy,fz_des) # get trajectory and time_bp
                    #### motion start ###
                    if not t_robot_thinking_flag:
                        break
                    mctrl.motion_start_routine(traj_q[0],traj_fz[0],hover_height,hover_height_close,lin_vel=controller_params['jogging_speed'])
                    if not t_robot_thinking_flag:
                        break
                    joint_force_exe, cart_force_exe = mctrl.trajectory_force_PIDcontrol(traj_xy,traj_q,traj_fz,force_lookahead=True)
                    if not t_robot_thinking_flag:
                        break
                    mctrl.motion_end_routine(traj_q[-1],hover_height, lin_vel=controller_params['jogging_speed'])
            except KeyboardInterrupt:
                break
            if not t_robot_thinking_flag:
                break
        
        #jog to end point
        mctrl.motion_end_routine(traj_q[-1],hover_height*4, lin_vel=controller_params['jogging_speed'])
        mctrl.params['lookahead_time'] = origin_lookaheadt
        mctrl.params['force_ctrl_damping'] = origin_fdamp
        mctrl.params["moveL_acc_lin"]  = origin_acc         
        print('Robot ready')

    # start robot thinking while waiting for user input
    t_robot_thinking_flag = True
    t_robot_thinking = threading.Thread(target=robot_thinking)
    t_robot_thinking.start()
    ## Next round activatoion
    # input_key = ''
    # while input_key!='n':
    #     input_key = input("Press n + Enter to start next round")
    messagebox.showinfo('Message', 'Next Round?')
    # stop robot thinking
    t_robot_thinking_flag = False
    t_robot_thinking.join()

    #jog to initial_position
    mctrl.jog_joint_position_cmd(q_tracking_start,v=controller_params['jogging_speed'],wait_time=0.5)

    ###################### Face tracking ######################
    if TAKE_FACE_IMAGE:
        print("FACE TRACKING")
        q_cmd_prev=q_tracking_start
        while True:
            loop_start_time=time.time()
            wire_packet=bbox_wire.TryGetInValue()
            
            
            q_cur=mctrl.read_position()
            pose_cur=robot_cam.fwd(q_cur)
            if mctrl.USE_RR_ROBOT:
                time.sleep(mctrl.TIMESTEP)
            
            if wire_packet[0]:
                bbox=wire_packet[1]
                if len(bbox)==0 or (q_cur[0]<angle_range[0] or q_cur[0]>angle_range[1]):
                    # if no face detected, or out of range
                    # jog to initial position
                    diff=q_tracking_start-q_cur
                    if np.linalg.norm(diff)>face_track_speed/2:
                        qdot=diff/np.linalg.norm(diff)
                    else:
                        qdot=diff
                else:	#if face detected
                    #calculate size of bbox
                    size=np.array([bbox[2]-bbox[0],bbox[3]-bbox[1]])
                    #calculate center of bbox
                    center=np.array([bbox[0]+size[0]/2,bbox[1]+size[1]/2])
                    y_gain=-1 # -0.8
                    x_gain=-0.001 # -0.08
                    yd=center[1]-image_center[1]
                    xd=center[0]-image_center[0]
                    try:
                        if pose_cur.p[2]<height_range[0]:
                            yd = min(0,yd)
                        elif pose_cur.p[2]>height_range[1]:
                            yd = max(0,yd)
                        q_temp=robot_cam.inv(pose_cur.p+yd*np.array([0,0,y_gain]),pose_cur.R,q_cur)[0]
                    except:
                        continue
                    q_temp+=xd*np.array([x_gain,0,0,0,0,0])
                    q_diff=q_temp-q_cur
                    if np.linalg.norm(q_diff)>face_track_speed:
                        qdot=face_track_speed*q_diff/np.linalg.norm(q_diff)
                    else:
                        qdot=q_diff
                    # vdot = (x_gain*xd*face_track_x + y_gain*yd*face_track_y)
                    # vdot = y_gain*yd*face_track_y
                    
                    # if np.linalg.norm(vdot)>face_track_speed:
                    #     vdot=face_track_speed*vdot/np.linalg.norm(vdot)
                    # J_mat = robot.jacobian(q_cur)
                    # qdot = np.linalg.pinv(J_mat)@np.append(np.zeros(3),vdot)
                    # qdot = qdot + np.array([xd*x_gain,0,0,0,0,0])
                    
                    if np.linalg.norm(qdot)<0.1:
                        # print(time.time()-start_time)
                        if time.time()-start_time>3:
                            break
                    else:
                        start_time=time.time()
                
                # send position command to robot
                q_cmd=q_cmd_prev+qdot*mctrl.TIMESTEP
                mctrl.position_cmd(q_cmd)
                q_cmd_prev=copy.deepcopy(q_cmd)

        ##### Get face image #####
        RR_image=image_wire.TryGetInValue()
        if RR_image[0]:
            img=RR_image[1]
            img=np.array(img.data,dtype=np.uint8).reshape((img.image_info.height,img.image_info.width,3))
            #get the image within the bounding box, a bit larger than the bbox
            img=img[int(bbox[1]-size[1]/5):int(bbox[3]+size[1]/9),int(bbox[0]-size[0]/9):int(bbox[2]+size[0]/9),:]
        print('IMAGE TAKEN')
        cv2.imwrite(TEMP_DATA_DIR+'img.jpg',img)
    else:
        img = cv2.imread(TEMP_DATA_DIR+'img.jpg')
        # plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        # plt.show()
        # img = cv2.imread(TEMP_DATA_DIR+'img_'+img_names[img_count%len(img_names)]+'.jpg')
        # print("Drawing image: ", img_names[img_count%len(img_names)])
        # img_count+=1

    print("PAGE FLIPPING")
    mctrl.press_button_routine(p_button,R_pencil,h_offset=hover_height,lin_vel=controller_params['jogging_speed'], q_seed=q_seed)

    img_st = time.time()
    # cv2.imshow('img',img)
    # cv2.waitKey(0)
    ############################################################

    ########################## portrait FaceSegmentation/GAN ##############################
    ## Face Segmentation
    if FACE_PORTRAIT:
        gray_image_masked,image_mask,face_mask,_ = faceseg.get_face_mask(img)
        anime_img = anime.forward(gray_image_masked)
        img_gray=cv2.cvtColor(anime_img, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(TEMP_DATA_DIR+'img_out.jpg',anime_img)
    else:
        anime_img = cv2.imread(TEMP_DATA_DIR+'img_out.jpg')
        img_gray=cv2.cvtColor(anime_img, cv2.COLOR_BGR2GRAY)

    # cv2.imshow('img',anime_img)
    # cv2.waitKey(0)
    print("IMAGE PROCESSING TIME: ", time.time()-img_st)
    ####################################################################
    
    ####################################PLANNING#####################################################
    planning_st = time.time()
    ###Pixel Traversal
    print('TRAVERSING PIXELS')    
    resize_ratio=np.mean(np.divide(target_size,anime_img.shape[:2]))
    if PIXEL_PLAN:
        # face_drawing_order=[10,1,6,(7,8,9),2,3,4,5,0]
        # pixel_paths, image_thresh = travel_pixel_dots(anime_img,resize_ratio=resize_ratio,max_radias=10,min_radias=2,face_mask=image_mask,face_drawing_order=face_drawing_order,SHOW_TSP=True)
        
        face_drawing_order=[10,1,(6,1),(7,8,9),(2,1),(3,1),(4,1),(5,1),0] # hair, face, nose, upper lip, teeth, lower lip, left eyebrow, right eyebrow, left eye, right eye
        pixel_paths, image_thresh = travel_pixel_skeletons(anime_img,resize_ratio=resize_ratio,max_radias=max_stroke_w,min_radias=min_stroke_w,face_mask=image_mask,face_drawing_order=face_drawing_order,SHOW_TSP=True)
        pickle.dump(pixel_paths, open(TEMP_DATA_DIR+'pixel_paths.pkl', 'wb'))
        cv2.imwrite(TEMP_DATA_DIR+'img_thresh.jpg',image_thresh)
    else:
        pixel_paths = pickle.load(open(TEMP_DATA_DIR+'pixel_paths.pkl', 'rb'))
        image_thresh = cv2.imread(TEMP_DATA_DIR+'img_thresh.jpg',0)
        
    print("Image size: ", image_thresh.shape)
    ###Project to IPAD
    print("PROJECTING TO IPAD")
    if CART_PLAN:
        _,cartesian_paths_world,force_paths=image2plane(image_thresh,ipad_pose,pixel2mm,pixel_paths,pixel2force)
        pickle.dump(cartesian_paths_world, open(TEMP_DATA_DIR+'cartesian_paths_world.pkl', 'wb'))
        pickle.dump(force_paths, open(TEMP_DATA_DIR+'force_paths.pkl', 'wb'))
    else:
        cartesian_paths_world = pickle.load(open(TEMP_DATA_DIR+'cartesian_paths_world.pkl', 'rb'))
        force_paths = pickle.load(open(TEMP_DATA_DIR+'force_paths.pkl', 'rb'))

    ###Solve Joint Trajectory
    print("SOLVING JOINT TRAJECTORY")
    if JS_PLAN:
        js_paths=[]
        for cartesian_path in cartesian_paths_world:
            curve_js=robot.find_curve_js(cartesian_path,[R_pencil_base]*len(cartesian_path),q_seed)
            js_paths.append(curve_js)
        pickle.dump(js_paths, open(TEMP_DATA_DIR+'js_paths.pkl', 'wb'))
    else:
        js_paths = pickle.load(open(TEMP_DATA_DIR+'js_paths.pkl', 'rb'))
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
            if lam[-1] < smallest_lam:
                continue
            traj_q, traj_xy, traj_fz, time_bp = mctrl.trajectory_generate(js_paths[i],curve_xy,fz_des) # get trajectory and time_bp
            #### motion start ###
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
    
    ########### Logo words ###########
    ####### write words
    num_segments=len(glob.glob('path/pixel_path/'+test_logo+'/*.csv'))
    img=cv2.imread('imgs/'+test_logo+'_resized.png')
    # plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # plt.show()
    # replocate the image
    offset_ratio=0.975
    img_new = np.ones((image_thresh.shape[0]+2*img.shape[0],image_thresh.shape[1]+2*img.shape[1],3))*255
    pixel_offset = np.array([(image_thresh.shape[1]+img.shape[1])*0.8,(image_thresh.shape[0]+img.shape[0])*(1-offset_ratio)]).astype(int)
    img_new[pixel_offset[1]:img.shape[0]+pixel_offset[1],pixel_offset[0]:pixel_offset[0]+img.shape[1]]=img
    img = img_new
    # cv2.imwrite(TEMP_DATA_DIR+'img_logo.png',img)
    # plt.imshow(img)
    # plt.show()
    # relocate paths
    pixel_paths=[]
    for i in range(num_segments):
        pixel_paths.append(np.loadtxt('path/pixel_path/'+test_logo+'/%i.csv'%i,delimiter=',').reshape((-1,3)))
        pixel_paths[-1][:,:2]+=pixel_offset
    # pickle.dump(pixel_paths, open(TEMP_DATA_DIR+'pixel_paths_logo.pkl', 'wb'))
    
    print("PROJECTING TO IPAD")
    _,cartesian_paths_world,force_paths=image2plane(img,ipad_pose,pixel2mm,pixel_paths,pixel2force)
    # pickle.dump(cartesian_paths_world, open(TEMP_DATA_DIR+'cartesian_paths_world_logo.pkl', 'wb'))
    
    print("SOLVING JOINT TRAJECTORY")
    js_paths=[]
    for cartesian_path in cartesian_paths_world:
        curve_js=robot.find_curve_js(cartesian_path,[R_pencil_base]*len(cartesian_path),q_seed)
        js_paths.append(curve_js)
    # pickle.dump(js_paths, open(TEMP_DATA_DIR+'js_paths_logo.pkl', 'wb'))
    ### Execute
    execution_st = time.time()
    
    print('START DRAWING LOGO')
    origin_movelspeed = mctrl.params['moveL_speed_lin']
    origin_acc = mctrl.params["moveL_acc_lin"] 
    origin_lookaheadt = mctrl.params['lookahead_time']
    origin_fdamp = mctrl.params['force_ctrl_damping']
    mctrl.params['lookahead_time'] = 0.04
    mctrl.params["moveL_acc_lin"] = origin_acc
    mctrl.params['force_ctrl_damping'] = origin_fdamp
    num_segments = len(js_paths)
    print("LOGO NUM SEGMENTS: ", num_segments)
    ###Execute
    for i in range(0,num_segments):
        if len(js_paths[i])<=1:
            continue
        cartesian_path_world = cartesian_paths_world[i]
        force_path = force_paths[i]
        curve_xyz = np.dot(mctrl.ipad_pose_inv[:3,:3],cartesian_path_world.T).T+np.tile(mctrl.ipad_pose_inv[:3,-1],(len(cartesian_path_world),1))
        curve_xy = curve_xyz[:,:2] # get xy curve
        fz_des = force_path*(-1) # transform to tip desired
        # fz_des = fz_des*pixelforce_ratio_calib
        lam = calc_lam_js(js_paths[i],mctrl.robot) # get path length
        if lam[-1] < smallest_lam:
            continue
        traj_q, traj_xy, traj_fz, time_bp = mctrl.trajectory_generate(js_paths[i],curve_xy,fz_des) # get trajectory and time_bp
        #### motion start ###
        mctrl.motion_start_routine(traj_q[0],traj_fz[0],hover_height,hover_height_close,lin_vel=controller_params['jogging_speed'])
        joint_force_exe, cart_force_exe = mctrl.trajectory_force_PIDcontrol(traj_xy,traj_q,traj_fz,force_lookahead=True)
        mctrl.motion_end_routine(traj_q[-1],hover_height, lin_vel=controller_params['jogging_speed'])

    #jog to end point
    mctrl.motion_end_routine(traj_q[-1],hover_height*4, lin_vel=controller_params['jogging_speed'])
    mctrl.params['lookahead_time'] = origin_lookaheadt
    mctrl.params['force_ctrl_damping'] = origin_fdamp
    mctrl.params["moveL_acc_lin"]  = origin_acc

    print('FINISHED DRAWING LOGO')
    #######################################
    
    
    print('TOTAL TIME: ', time.time()-img_st)
    messagebox.showinfo('Message', 'Next Round?')