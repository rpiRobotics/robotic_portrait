from RobotRaconteur.Client import *
import numpy as np
import matplotlib.pyplot as plt
import glob, sys, time
from scipy.interpolate import CubicSpline
from general_robotics_toolbox import *
from abb_robot_client.egm import EGM
import abb_motion_program_exec as abb
from copy import deepcopy
from pathlib import Path

sys.path.append('../toolbox')
from robot_def import *
from lambda_calc import *
from motion_toolbox import *

SHOW_STATUS = False

def spline_js(cartesian_path,curve_js,vd,rate=250):
    lam=calc_lam_cs(cartesian_path)
    # polyfit=np.polyfit(lam,curve_js,deg=40)
    # lam=np.linspace(0,lam[-1],int(rate*lam[-1]/vd))
    # return np.vstack((np.poly1d(polyfit[:,0])(lam), np.poly1d(polyfit[:,1])(lam), np.poly1d(polyfit[:,2])(lam), np.poly1d(polyfit[:,3])(lam), np.poly1d(polyfit[:,4])(lam), np.poly1d(polyfit[:,5])(lam))).T
    polyfit=CubicSpline(lam,curve_js)
    lam=np.linspace(0,lam[-1],int(rate*lam[-1]/vd))
    return polyfit(lam)

def adjoint_map(T):
    R=T.R
    p=T.p
    return np.vstack((np.hstack((R,np.zeros((3,3)))),np.hstack((hat(p)@R,R))))

class MotionController(object):
    def __init__(self,robot,ipad_pose,H_pentip2ati,controller_param,TIMESTEP,USE_RR_ROBOT=False,
                 RR_robot_sub=None,FORCE_PROTECTION=5,RR_ati_cli=None,simulation=False,abb_robot_ip=''):
        
        # define robots
        self.robot=robot
        # pose relationship
        self.ipad_pose=ipad_pose
        self.ipad_pose_inv = np.linalg.inv(self.ipad_pose)
        self.ipad_pose_T = Transform(self.ipad_pose[:3,:3],self.ipad_pose[:3,-1])
        self.ipad_pose_inv_T = self.ipad_pose_T.inv()
        self.H_pentip2ati=H_pentip2ati
        self.H_ati2pentip=np.linalg.inv(H_pentip2ati)
        self.ad_ati2pentip=adjoint_map(Transform(self.H_ati2pentip[:3,:3],self.H_ati2pentip[:3,-1]))
        self.ad_ati2pentip_T=self.ad_ati2pentip.T
        # controller parameters
        self.params=controller_param
        # Force protection. Units: N
        self.FORCE_PROTECTION = FORCE_PROTECTION

        if RR_ati_cli is not None and not simulation:
            #################### FT Connection ####################
            self.RR_ati_cli=RR_ati_cli
            #Connect a wire connection
            wrench_wire = self.RR_ati_cli.wrench_sensor_value.Connect()
            #Add callback for when the wire value change
            wrench_wire.WireValueChanged += self.wrench_wire_cb
        
        ################ Robot Connection ################
        self.USE_RR_ROBOT=USE_RR_ROBOT
        self.command_seqno = 1
        if not simulation:
            if self.USE_RR_ROBOT and RR_robot_sub is not None:
                ##RR PARAMETERS
                self.RR_robot_sub=RR_robot_sub
                # RR_robot_sub=RRN.SubscribeService('rr+tcp://localhost:58655?service=robot')
                self.RR_robot=self.RR_robot_sub.GetDefaultClientWait(1)
                self.robot_state = self.RR_robot_sub.SubscribeWire("robot_state")
                time.sleep(0.1)
                robot_const = RRN.GetConstants("com.robotraconteur.robotics.robot", self.RR_robot)
                self.halt_mode = robot_const["RobotCommandMode"]["halt"]
                self.position_mode = robot_const["RobotCommandMode"]["position_command"]
                self.RobotJointCommand = RRN.GetStructureType("com.robotraconteur.robotics.robot.RobotJointCommand",self.RR_robot)
                self.RR_robot.command_mode = self.halt_mode
                time.sleep(0.1)
                self.connect_position_mode()
            else: # use EGM else
                self.client = abb.MotionProgramExecClient(base_url="http://"+abb_robot_ip+":80")
                self.egm = EGM()
  
        self.TIMESTEP = TIMESTEP # egm timestep 4 ms, ur5 10 ms
        self.params['lookahead_index'] = int(self.params['lookahead_time']/self.TIMESTEP)

        self.ft_reading = None
        if not simulation:
            while self.ft_reading is None:
                time.sleep(0.1)
        print("Motion controller initialized")
        
    def connect_position_mode(self):
     
        ## connect to position command mode wire
        self.RR_robot.command_mode = self.position_mode
        self.cmd_w = self.RR_robot_sub.SubscribeWire("position_command")
    
    def position_cmd(self,q):

        # Increment command_seqno
        self.command_seqno += 1
        
        if self.USE_RR_ROBOT:
            # Create Fill the RobotJointCommand structure
            joint_cmd = self.RobotJointCommand()
            joint_cmd.seqno = self.command_seqno # Strictly increasing command_seqno
            joint_cmd.state_seqno = self.robot_state.InValue.seqno # Send current robot_state.seqno as failsafe
            # Set the joint command
            joint_cmd.command = q
            # Send the joint command to the robot
            self.cmd_w.SetOutValueAll(joint_cmd)
        else:
            self.egm.send_to_robot(np.degrees(q))

    def start_egm(self):
        mm = abb.egm_minmax(-1e-3,1e-3)
        egm_config = abb.EGMJointTargetConfig(mm,mm,mm,mm,mm,mm,1000,1000)
        mp = abb.MotionProgram(egm_config=egm_config)
        mp.EGMRunJoint(10,0.05,0.05)
        lognum = self.client.execute_motion_program(mp,wait=False)

    def stop_egm(self):
        self.client.stop_egm()
    
    def read_position(self):
        if self.USE_RR_ROBOT:
            return self.robot_state.InValue.joint_position
        else:
            res, state = self.egm.receive_from_robot(timeout=0.1)
            if not res:
                raise Exception("Robot communication lost")
            return np.radians(state.joint_angles)
    
    def drain_egm(self, read_N):

        for i in range(read_N):
            self.read_position()
    
    def wrench_wire_cb(self,w,value,t):

        ft_reading_now = np.array([value['torque']['x'],value['torque']['y'],value['torque']['z'],value['force']['x'],value['force']['y'],value['force']['z']])
        if self.ft_reading is None:
            self.ft_reading = ft_reading_now
        else:
            self.ft_reading = self.ft_reading*(1-self.params['force_filter_alpha'])+ft_reading_now*self.params['force_filter_alpha']
        self.last_ft_time = time.time()

    def force_impedence_ctrl(self,f_err):
     
        return self.params["force_ctrl_damping"]*f_err

    def force_load_z(self,fz_des,load_speed=None):

        if load_speed is None:
            load_speed = self.params['load_speed']

        touch_t = None
        this_st=None
        set_time=None
        ft_record=[]
        while True:
            # tool pose reading
            q_now = self.read_position()
            tip_now = self.robot.fwd(q_now)
            tip_now_ipad = self.ipad_pose_T.inv()*tip_now
            # force reading
            ft_tip = self.ad_ati2pentip_T@self.ft_reading
            fz_now = float(ft_tip[-1])
            # force protection
            if np.linalg.norm(ft_tip[3:])>self.FORCE_PROTECTION:
                print("force: ",ft_tip[3:])
                print("force too large")
                break
            if time.time()-self.last_ft_time>0.1:
                print("force reading lost")
                break
            # time force joint angle record
            ft_record.append(np.append(np.array([time.time(),fz_now]),np.degrees(q_now)))
            # force control
            tip_next_ipad = deepcopy(tip_now_ipad)
            if np.abs(fz_now) < self.params['force_epsilon'] and touch_t is None: # if not touch ipad
                tip_next_ipad.p[2] = tip_now_ipad.p[2] + -1*load_speed * self.TIMESTEP # move in -z direction
            else: # when touch ipad
                if touch_t is None:
                    touch_t=time.time()
                # track a trapziodal force profile
                this_fz_des = min(fz_des,(time.time()-touch_t)*self.params["trapzoid_slope"])
                f_err = this_fz_des-fz_now # feedback error
                v_des = self.force_impedence_ctrl(f_err) # force control
                tip_next_ipad.p[2] = tip_now_ipad.p[2] + v_des * self.TIMESTEP # force impedence control 
            
            # check if force achieved
            if np.fabs(fz_des-fz_now)<self.params['force_epsilon']:
                if set_time is None:
                    set_time = time.time()
                if (time.time()-set_time)>self.params['settling_time']:
                    break
            else:
                set_time = time.time()

            # get joint angles using ik
            tip_next = self.ipad_pose_T*tip_next_ipad
            # print(tip_next.p)
            q_des = self.robot.inv(tip_next.p,tip_next.R,q_now)[0]
            
            # send robot position command
            # if this_st is not None:
            # 	while time.time()-this_st<self.TIMESTEP-0.00001:
            # 		pass
            # this_st=time.time()
            self.position_cmd(q_des)
        
        return ft_record
   
    def motion_start_routine(self, js_start, f_start, h_offset, h_offset_low, lin_vel=2):
        
        # jog to start point
        pose_start=self.robot.fwd(js_start)
        #arc-like trajectory to next segment
        p_start=pose_start.p+h_offset*self.ipad_pose[:3,-2]
        q_start=self.robot.inv(p_start,pose_start.R,js_start)[0]
        pose_cur=self.robot.fwd(self.read_position())
        p_mid=(pose_start.p+pose_cur.p)/2+h_offset*self.ipad_pose[:3,-2]
        q_mid=self.robot.inv(p_mid,pose_start.R,js_start)[0]
        self.trajectory_position_cmd(np.vstack((self.read_position(),q_mid,q_start)),v=lin_vel)
        p_start=pose_start.p+h_offset_low*self.ipad_pose[:3,-2]
        q_start=self.robot.inv(p_start,pose_start.R,js_start)[0]
        self.jog_joint_position_cmd(q_start,v=lin_vel,wait_time=0.5)
        # set tare before load force
        self.RR_ati_cli.setf_param("set_tare", RR.VarValue(True, "bool"))
        if SHOW_STATUS:
            input("Set tare. Start load force?")
        # print("Start load force")
        # load force
        ft_record=self.force_load_z(f_start)
        # print("Load force done")
        
        return ft_record
    
    def motion_end_routine(self, js_end, h_offset, lin_vel=2):
        # jog to end point z+hoffset
        pose_end=self.robot.fwd(js_end)
        p_end=pose_end.p+h_offset*self.ipad_pose[:3,-2]
        q_end=self.robot.inv(p_end,pose_end.R,js_end)[0]
        self.jog_joint_position_cmd(q_end,v=lin_vel,wait_time=0.5)
    
    def press_button_routine(self, p_button, R_button, h_offset=30, lin_vel=2, press_force=1, q_seed=None):
        
        if q_seed is None:
            q_seed=self.read_position() 
        # jog to button point
        T_button_offset_robot=self.ipad_pose_T*Transform(R_button,p_button+np.array([0,0,h_offset]))
        T_button_robot=self.ipad_pose_T*Transform(R_button,p_button+np.array([0,0,1]))
        q_button_offset=self.robot.inv(T_button_offset_robot.p,T_button_offset_robot.R,q_seed)[0]
        q_button=self.robot.inv(T_button_robot.p,T_button_robot.R,q_seed)[0]
        self.trajectory_position_cmd(np.vstack((self.read_position(),q_button_offset,q_button)),v=lin_vel) # move about the button
        # self.jog_joint_position_cmd(q_button_offset,v=lin_vel) 
        # self.jog_joint_position_cmd(q_button,v=lin_vel) # move about the button
        self.RR_ati_cli.setf_param("set_tare", RR.VarValue(True, "bool"))
        self.force_load_z(-press_force,load_speed=lin_vel) # press the button
        self.jog_joint_position_cmd(q_button_offset,v=lin_vel) # move about the button
    
    def trajectory_generate(self,curve_js,curve_xy,force_path,lin_vel=None,lin_acc=None):
    
        # velocity and acceleration
        if lin_vel is None:
            lin_vel = self.params['moveL_speed_lin']
        if lin_acc is None:
            lin_acc = self.params['moveL_acc_lin']
        # Calculate the path length
        lam = calc_lam_js(curve_js, self.robot)
        
        # find the time for each segment, with acceleration and deceleration
        if len(lam)>2 and lin_acc>0:
            time_bp = np.zeros_like(lam)
            acc = lin_acc
            vel = 0
            for i in range(0,len(lam)):
                if vel>=lin_vel:
                    time_bp[i] = time_bp[i-1]+(lam[i]-lam[i-1])/lin_vel
                else:
                    time_bp[i] = np.sqrt(2*lam[i]/acc)
                    vel = acc*time_bp[i]
            time_bp_half = []
            vel = 0
            for i in range(len(lam)-1,-1,-1):
                if vel>=lin_vel or i<=len(lam)/2:
                    break
                else:
                    time_bp_half.append(np.sqrt(2*(lam[-1]-lam[i])/acc))
                    vel = acc*time_bp_half[-1]
            time_bp_half = np.array(time_bp_half)[::-1]
            time_bp_half = time_bp_half*-1+time_bp_half[0]
            time_bp[-len(time_bp_half):] = time_bp[-len(time_bp_half)-1]+time_bp_half\
                +(lam[-len(time_bp_half)]-lam[-len(time_bp_half)-1])/lin_vel
            if SHOW_STATUS:
                plt.plot(time_bp,lam)
                plt.show()
            ###
        else:
            time_bp = lam/lin_vel
        
        # total time stamps
        time_stamps = np.arange(0,time_bp[-1],self.TIMESTEP)
        time_stamps = np.append(time_stamps,time_bp[-1])

        # polyfit=CubicSpline(time_bp,curve_js, bc_type='natural')
        # traj_q_p = polyfit(time_stamps)
        # polyfit=CubicSpline(time_bp,curve_xy, bc_type='natural')
        # traj_xy_p = polyfit(time_stamps)
        # polyfit=CubicSpline(time_bp,force_path, bc_type='natural')
        # traj_fz_p = polyfit(time_stamps)


        # Calculate the number of steps for the trajectory
        num_steps = int(time_bp[-1] / self.TIMESTEP)

        # Initialize the trajectory list
        traj_q = []
        traj_xy = []
        traj_fz = []

        # Generate the trajectory
        for step in range(num_steps):
            # Calculate the current time
            current_time = step * self.TIMESTEP
            
            # Find the current segment
            for i in range(len(time_bp)-1):
                if current_time >= time_bp[i] and current_time < time_bp[i+1]:
                    seg = i
                    break
            
            # Calculate the fraction of time within the current segment
            frac = (current_time - time_bp[seg]) / (time_bp[seg+1] - time_bp[seg])
            
            # Calculate the desired joint position for the current step
            q_des = frac * curve_js[seg+1] + (1 - frac) * curve_js[seg]
            # Calculate the desired position for the current step
            xy_des = frac * curve_xy[seg+1] + (1 - frac) * curve_xy[seg]
            # Calculate the desired force for the current step
            fz_des = frac * force_path[seg+1] + (1 - frac) * force_path[seg]
            
            # Append the desired position to the trajectory
            traj_q.append(q_des)
            traj_xy.append(xy_des)
            traj_fz.append(fz_des)
        
        
        # if force_path[0]!=0:
        #     plt.plot(traj_q[1])
        #     plt.plot(traj_q_p[1])
        #     plt.title("Trajectory q")
        #     plt.show()

        #     plt.plot(traj_fz)
        #     plt.plot(traj_fz_p)
        #     plt.title("Trajectory fz")
        #     plt.show()

        # traj_fz=np.array(traj_fz_p)
        # traj_q=np.array(traj_q_p)
        # traj_xy=np.array(traj_xy_p)
        
        return np.array(traj_q), np.array(traj_xy), np.array(traj_fz), np.array(time_bp)
    
    def trajectory_force_PIDcontrol(self,traj_xy,traj_js,traj_fz,force_lookahead=False):
        
        assert len(traj_xy)==len(traj_fz), "trajectory length mismatch"
        assert len(traj_xy)==len(traj_js), "trajectory length mismatch"
        
        ### trajectory force control
        start_time=time.time()
        joint_force_exe=[]
        cart_force_exe=[]
        force_tracked=None
        for i in range(len(traj_xy)):
            # joint reading
            q_now = self.read_position()
            tip_now = self.robot.fwd(q_now)
            tip_now_ipad = self.ipad_pose_inv_T*tip_now
            # read force
            # ft_tip = self.ad_ati2pentip_T@self.ft_reading
            ft_tip = self.ad_ati2pentip_T@np.append(np.zeros(3),self.ft_reading[3:])
            fz_now = float(ft_tip[-1])
            # fz_now = float(self.ft_reading[-1])
            force_tracked = fz_now
            if np.linalg.norm(ft_tip[3:])>self.FORCE_PROTECTION: # force protection break
                print("force: ",ft_tip[3:])
                print("force too large")
                break
            if time.time()-self.last_ft_time>0.1: # force reading lost break
                print("force reading lost")
                break
            ### force feedback control
            # get current desired force
            if force_lookahead:
                fz_des = traj_fz[min(i+self.params['lookahead_index'],len(traj_fz)-1)]
            else:
                fz_des = traj_fz[i]
            # f_err = fz_des-fz_now # feedback error lookahead
            f_err = fz_des-force_tracked # feedback error lookahead
            v_des_z = self.force_impedence_ctrl(f_err) # force control
            # get xyz (in ipad frame) next
            next_T = self.ipad_pose_inv_T*self.robot.fwd(traj_js[i])
            next_T.p = np.append(traj_xy[i],tip_now_ipad.p[2]+v_des_z*self.TIMESTEP)
            next_T_world = self.ipad_pose_T*next_T
            # get desired joint angles using ik
            q_des = self.robot.inv(next_T_world.p,next_T_world.R,q_now)[0]

            ### send robot position command
            self.position_cmd(q_des)
            
            # time, force, joint angle, record
            joint_force_exe.append(np.append(np.array([time.time()-start_time,force_tracked]),q_now))
            # time, force, xyz, record
            cart_force_exe.append(np.append(np.array([time.time()-start_time,force_tracked]),tip_now_ipad.p))
            
            if self.USE_RR_ROBOT:
                time.sleep(self.TIMESTEP)
            
        return np.array(joint_force_exe), np.array(cart_force_exe)
    
    def jog_joint_position_cmd(self,q,v=0.4,wait_time=0):

        q_start=self.read_position()
        # total_time=np.linalg.norm(q-q_start)/v

        q_all = np.linspace(q_start,q,num=100)
        # print(q_all[:10])
        # print(q_all[-10:])
        # exit()
        # q_all = np.vstack((q_start,q))
        traj_q, traj_xy, traj_fz, time_bp=self.trajectory_generate(q_all,np.zeros(len(q_all)),np.zeros(len(q_all)),lin_vel=v,lin_acc=self.params["jogging_acc"])

        for i in range(len(traj_q)):
            self.read_position()
            self.position_cmd(traj_q[i])
            if self.USE_RR_ROBOT:
                time.sleep(self.TIMESTEP)
            
        ###additional points for accuracy
        start_time=time.time()
        while time.time()-start_time<wait_time:
            self.read_position()
            self.position_cmd(q)
    
    def jog_joint_position_cmd_nowait(self,q,v=0.4,wait_time=0):

        q_start=self.read_position()
        # total_time=np.linalg.norm(q-q_start)/v

        q_all = np.linspace(q_start,q,num=100)
        # print(q_all[:10])
        # print(q_all[-10:])
        # exit()
        # q_all = np.vstack((q_start,q))
        traj_q, traj_xy, traj_fz, time_bp=self.trajectory_generate(q_all,np.zeros(len(q_all)),np.zeros(len(q_all)),lin_vel=v,lin_acc=self.params["jogging_acc"])

        for i in range(len(traj_q)):
            self.position_cmd(traj_q[i])
            time.sleep(self.TIMESTEP)
            
        ###additional points for accuracy
        start_time=time.time()
        while time.time()-start_time<wait_time:
            self.position_cmd(q)
            time.sleep(self.TIMESTEP)
    
    def trajectory_position_cmd(self,q_all,v=0.4,wait_time=0):

        if len(q_all)<100:
            iter_num = int(100/(len(q_all)-1))
            q_smooth_all=[]
            for qi in range(len(q_all)-1):
                q_smooth_all.extend(np.linspace(q_all[qi],q_all[qi+1],num=iter_num,endpoint=False))
            q_smooth_all.append(q_all[-1])
            q_all=q_smooth_all

        traj_q, traj_xy, traj_fz, time_bp=self.trajectory_generate(q_all,np.zeros(len(q_all)),np.zeros(len(q_all)),lin_vel=v,lin_acc=self.params["jogging_acc"])

        for i in range(len(traj_q)):
            self.read_position()
            self.position_cmd(traj_q[i])
            if self.USE_RR_ROBOT:
                time.sleep(self.TIMESTEP)
        
        ###additional points for accuracy
        start_time=time.time()
        while time.time()-start_time<wait_time:
            self.position_cmd(traj_q[-1])
            time.sleep(self.TIMESTEP)
    
    def trajectory_position_cmd_nowait(self,q_all,v=0.4):

        if len(q_all)<100:
            iter_num = int(100/(len(q_all)-1))
            q_smooth_all=[]
            for qi in range(len(q_all)-1):
                q_smooth_all.extend(np.linspace(q_all[qi],q_all[qi+1],num=iter_num,endpoint=False))
            q_smooth_all.append(q_all[-1])
            q_all=q_smooth_all

        traj_q, traj_xy, traj_fz, time_bp=self.trajectory_generate(q_all,np.zeros(len(q_all)),np.zeros(len(q_all)),lin_vel=v,lin_acc=self.params["jogging_acc"])

        for i in range(len(traj_q)):
            # self.read_position()
            self.position_cmd(traj_q[i])
            time.sleep(self.TIMESTEP)