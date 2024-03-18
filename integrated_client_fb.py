from RobotRaconteur.Client import *     #import RR client library
import time, traceback, sys, cv2
import numpy as np
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
FORCE_FEEDBACK=False

if ROBOT_NAME=='ABB_1200_5_90':
    #########################################################config parameters#########################################################
    robot_cam=robot_obj(ROBOT_NAME,'config/ABB_1200_5_90_robot_default_config.yml',tool_file_path='config/camera.csv')
    robot=robot_obj(ROBOT_NAME,'config/ABB_1200_5_90_robot_default_config.yml',tool_file_path='config/heh6_pen2.csv')
    radius=500 ###eef position to robot base distance w/o z height
    angle_range=np.array([-3*np.pi/4,-np.pi/4]) ###angle range of joint 1 for robot to move
    height_range=np.array([500,900]) ###height range for robot to move
    p_start=np.array([0,-radius,700])	###initial position
    R_start=np.array([	[0,1,0],
                        [0,0,-1],
                        [-1,0,0]])	###initial orientation
    q_start=robot_cam.inv(p_start,R_start,np.zeros(6))[0]	###initial joint position
    image_center=np.array([1080,1080])/2	###image center
    RR_robot_sub=RRN.SubscribeService('rr+tcp://localhost:58651?service=robot')
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

RR_ati_cli=None
if FORCE_FEEDBACK:
    RR_ati_cli=RRN.ConnectService('rr+tcp://localhost:59823?service=ati_sensor') # connect to ATI sensor
#########################################################config parameters#########################################################
paper_size=np.loadtxt('config/paper_size.csv',delimiter=',') # size of the paper
pixel2mm=np.loadtxt('config/pixel2mm.csv',delimiter=',') # pixel to mm ratio
pixel2force=np.loadtxt('config/pixel2force.csv',delimiter=',') # pixel to force ratio
ipad_pose=np.loadtxt('config/ipad_pose.csv',delimiter=',') # ipad pose
H_pentip2ati=np.loadtxt('config/pentip2ati.csv',delimiter=',') # FT sensor info
p_button=[] # button position
hover_height=20 # height to hover above the paper
######## Controller parameters ###
controller_params = {
    "force_ctrl_damping": 60.0, # 180, 90, 60
    "force_epsilon": 0.1, # Unit: N
    "moveL_speed_lin": 6.0, # Unit: mm/sec
    "moveL_acc_lin": 1.0, # Unit: mm/sec^2
    "moveL_speed_ang": np.radians(10), # Unit: rad/sec
    "trapzoid_slope": 1, # trapzoidal load profile. Unit: N/sec
    "load_speed": 10.0, # Unit mm/sec
    "unload_speed": 1.0, # Unit mm/sec
    'settling_time': 1, # Unit: sec
    "lookahead_time": 0.132, # Unit: sec
    "jogging_speed": 10 # Unit: mm/sec
    }
### Define the motion controller
mctrl=MotionController(robot,ipad_pose,H_pentip2ati,controller_params,TIMESTEP,USE_RR_ROBOT=True,
                 RR_robot_sub=RR_robot_sub,FORCE_PROTECTION=5,RR_ati_cli=RR_ati_cli)

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

#########################################################EXECUTION#########################################################
while True:
    start_time=time.time()
    #jog to initial_position
    mctrl.jog_joint_position_cmd(q_tracking_start,v=controller_params['jogging_speed'],wait_time=0.5)

    ###################### Face tracking ######################
    q_cmd_prev=q_tracking_start
    while True:
        loop_start_time=time.time()
        wire_packet=bbox_wire.TryGetInValue()
        
        q_cur=mctrl.read_position()
        if mctrl.USE_RR_ROBOT:
            time.sleep(mctrl.TIMESTEP)

        if wire_packet[0]:
            bbox=wire_packet[1]
            if len(bbox)==0: #if no face detected, jog to initial position
                diff=q_tracking_start-q_cur
                if np.linalg.norm(diff)>0.1:
                    qdot=diff/np.linalg.norm(diff)
                else:
                    qdot=diff
            else:	#if face detected
                pose_cur=robot_cam.fwd(q_cur)
                if q_cur[0]<angle_range[0] or q_cur[0]>angle_range[1] or pose_cur.p[2]<height_range[0] or pose_cur.p[2]>height_range[1]:
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
                    q_temp=robot_cam.inv(pose_cur.p+zd*np.array([0,0,z_gain]),pose_cur.R,q_cur)[0]
                except:
                    continue
                q_temp+=xd*np.array([x_gain,0,0,0,0,0])
                q_diff=q_temp-q_cur
                if np.linalg.norm(q_diff)>0.8:
                    qdot=0.8*q_diff/np.linalg.norm(q_diff)
                else:
                    qdot=q_diff
                
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
    cv2.imwrite('img.jpg',img)
    ############################################################

    ########################## portrait FaceSegmentation/GAN ##############################
    ## Face Segmentation
    image_mask = faceseg.forward_faceonly(img)
    #convert dark pixels to bright pixels
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_image_masked = cv2.bitwise_and(gray_image, gray_image, mask = image_mask)
    # get second masked value (background) mask must be inverted
    background = np.full(gray_image.shape, 255, dtype=np.uint8)
    bk = cv2.bitwise_and(background, background, mask=cv2.bitwise_not(image_mask))
    gray_image_masked = cv2.add(gray_image_masked, bk)
    anime_img = anime.forward(gray_image_masked)
    img_gray=cv2.cvtColor(anime_img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite('img_out.jpg',anime_img)
    ####################################################################
    
    ####################################PLANNING#####################################################
    ###Pixel Traversal
    print('TRAVERSING PIXELS')
    pixel_paths, image_thresh = travel_pixel_dots(anime_img,resize_ratio=2,max_radias=10,min_radias=2)
    ###Project to IPAD
    print("PROJECTING TO IPAD")
    _,cartesian_paths_world,force_paths=image2plane(image_thresh,ipad_pose,pixel2mm,pixel_paths,pixel2force)

    ###Solve Joint Trajectory
    print("SOLVING JOINT TRAJECTORY")
    R_pencil=ipad_pose[:3,:3]@Ry(np.pi)
    js_paths=[]
    for cartesian_path in cartesian_paths_world:
        curve_js=robot.find_curve_js(cartesian_path,[R_pencil]*len(cartesian_path),q_seed)
        js_paths.append(curve_js)

    print("PAGE FLIPPING")
    mctrl.press_button_routine(p_button,R_pencil,h_offset=hover_height,lin_vel=controller_params['jogging_speed'], q_seed=q_seed)

    print('START DRAWING')
    num_segments = len(js_paths)
    ###Execute
    for i in range(0,num_segments):
        if len(curve_js)<=1:
            continue
        cartesian_path_world = cartesian_paths_world[i]
        force_path = force_paths[i]
        curve_xyz = np.dot(mctrl.ipad_pose_inv[:3,:3],cartesian_path_world.T).T+np.tile(mctrl.ipad_pose_inv[:3,-1],(len(cartesian_path_world),1))
        curve_xy = curve_xyz[:,:2] # get xy curve
        fz_des = force_path*(-1) # transform to tip desired
        lam = calc_lam_js(curve_js,mctrl.robot) # get path length
        traj_q, traj_xy, traj_fz, time_bp = mctrl.trajectory_generate(curve_js,curve_xy,fz_des) # get trajectory and time_bp
        #### motion start ###
        mctrl.motion_start_procedure(traj_q[0],traj_fz[0],hover_height,1,lin_vel=controller_params['jogging_speed'])
        joint_force_exe, cart_force_exe = mctrl.trajectory_force_PIDcontrol(traj_xy,traj_q,traj_fz,force_lookahead=True)
        mctrl.motion_end_procedure(traj_q[-1],hover_height, lin_vel=controller_params['jogging_speed'])

    #jog to end point
    mctrl.motion_end_procedure(traj_q[-1],hover_height*4, lin_vel=controller_params['jogging_speed'])

    print('FINISHED DRAWING')