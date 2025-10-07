import cv2
import sys
import numpy as np
import yaml
import time
import general_robotics_toolbox as rox
import threading
import RobotRaconteurCompanion as RRC
from RobotRaconteurCompanion.Util.SensorDataUtil import SensorDataUtil
from RobotRaconteur.Client import * 

import RobotRaconteur as RR
RRN=RR.RobotRaconteurNode.s

class ArucoDetector(object):
    def __init__(self) -> None:
        
        ### connect to face track service for image stream ###
        url='rr+tcp://localhost:52222/?service=Face_tracking'
        self.face_tracking_sub=RRN.SubscribeService(url)
        self.face_obj = self.face_tracking_sub.GetDefaultClientWait(1)		#connect, timeout=30s
        self.image_wire=self.face_tracking_sub.SubscribeWire("frame_stream")
        self.face_tracking_sub.ClientConnectFailed += self.connect_failed
        
        ### fiducial data type ###
        self._fiducials=RRN.GetStructureType('com.robotraconteur.fiducial.RecognizedFiducials')
        self._fiducial=RRN.GetStructureType('com.robotraconteur.fiducial.RecognizedFiducial')
        self._fiducials_sensor_data=RRN.GetStructureType('com.robotraconteur.fiducial.FiducialSensorData')
        self._namedposecovtype = RRN.GetStructureType('com.robotraconteur.geometry.NamedPoseWithCovariance')
        self._namedposetype = RRN.GetStructureType('com.robotraconteur.geometry.NamedPose')
        self._posetype = RRN.GetNamedArrayDType('com.robotraconteur.geometry.Pose')
        self._sensordatatype = RRN.GetStructureType('com.robotraconteur.sensordata.SensorDataHeader')
        self._tstype = RRN.GetPodDType('com.robotraconteur.datetime.TimeSpec2')
        self._sensor_data_util = SensorDataUtil(RRN)
        
        ### parameters for aruco detection ###
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_1000)
        self.aruco_params = cv2.aruco.DetectorParameters()
        camera_param = yaml.safe_load(open("../config/camera_parameters.yaml"))
        f_x = camera_param['camera_matrix']['fx']
        f_y = camera_param['camera_matrix']['fy']
        c_x = camera_param['camera_matrix']['cx']
        c_y = camera_param['camera_matrix']['cy']
        k1 = camera_param['distortion_coefficients']['k1']
        k2 = camera_param['distortion_coefficients']['k2']
        p1 = camera_param['distortion_coefficients']['p1']
        p2 = camera_param['distortion_coefficients']['p2']
        k3 = camera_param['distortion_coefficients']['k3']
        # camera matrix
        self.camera_matrix = np.array([[f_x, 0, c_x],
                                [0, f_y, c_y],
                                [0,  0,  1]], dtype=np.float32)
        # distortion coefficients
        self.dist_coeffs = np.array([k1, k2, p1, p2, k3], dtype=np.float32)
        print(self.camera_matrix)
        print(self.dist_coeffs)
        # rate
        self.rate = 30
        
        ### others
        self.SHOW_VIZ = True
    
    def connect_failed(s, client_id, url, err):
        print ("Client connect failed: " + str(client_id.NodeID) + " url: " + str(url) + " error: " + str(err))
        
    def start_detector(self):
        self._running = True
        self.seqno = 0
        self._thread = threading.Thread(target=self._run)
        self._thread.start()
    
    def stop_detector(self):
        self._running = False
        self._thread.join()
    
    def get_aruco_pose(self):
        
        RR_image = self.image_wire.TryGetInValue()
        if RR_image[0]:
            timestamp = time.time()
            image=RR_image[1]
            image=np.array(image.data,dtype=np.uint8).reshape((image.image_info.height,image.image_info.width,3))
            # Convert the image to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # Detect the ArUco markers
            corners, ids, rejected = cv2.aruco.detectMarkers(gray, self.aruco_dict, parameters=self.aruco_params)
            # Draw the detected markers on the image
            cv2.aruco.drawDetectedMarkers(image, corners, ids)
            # Estimate the pose of the detected markers
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, 0.091, self.camera_matrix, self.dist_coeffs)
            R_list=[]
            t_list=[]
            image_viz=None
            # Draw the pose axes on the image
            if rvecs is not None and tvecs is not None:
                for rvec, tvec in zip(rvecs, tvecs):
                    R_list.append(cv2.Rodrigues(rvec)[0])
                    t_list.append(tvec[0])
                if self.SHOW_VIZ:
                    image_viz = image.copy()
                    cv2.drawFrameAxes(image_viz, self.camera_matrix, self.dist_coeffs, rvec, tvec, 0.1)
            return R_list, t_list, ids, image, timestamp, image_viz
        else:
            return None, None, None, None, None, None
    
    def _run(self):
        # clear previous list
        fiducials = self._fiducials()
        fiducials.recognized_fiducials=[]
        
        while self._running:
            R_list, t_list, ids, image, stamp, image_viz = self.get_aruco_pose()
            
            if image_viz is None:
                time.sleep(1/self.rate)
                continue

            ## get rigid body
            for i in range(len(t_list)):
                rec_fiducials = self._fiducial()
                rec_fiducials.fiducial_marker = 'marker'+str(int(ids[i][0]))
                rec_fiducials.pose = self._namedposecovtype()
                rec_fiducials.pose.pose = self._namedposetype()
                rec_fiducials.pose.pose.pose = np.zeros((1,),dtype=self._posetype)
                rec_fiducials.pose.pose.pose[0]['position']['x'] = t_list[i][0]*1000 ## mm
                rec_fiducials.pose.pose.pose[0]['position']['y'] = t_list[i][1]*1000 ## mm
                rec_fiducials.pose.pose.pose[0]['position']['z'] = t_list[i][2]*1000 ## mm
                quat = rox.R2q(R_list[i])
                rec_fiducials.pose.pose.pose[0]['orientation']['w'] = quat[0]
                rec_fiducials.pose.pose.pose[0]['orientation']['x'] = quat[1]
                rec_fiducials.pose.pose.pose[0]['orientation']['y'] = quat[2]
                rec_fiducials.pose.pose.pose[0]['orientation']['z'] = quat[3]
                fiducials.recognized_fiducials.append(rec_fiducials)

            fiducials_sensor_data = self._fiducials_sensor_data()
            fiducials_sensor_data.sensor_data = self._sensordatatype()
            fiducials_sensor_data.sensor_data.seqno = int(self.seqno)
            nanosec = stamp*1e9
            fiducials_sensor_data.sensor_data.ts = np.zeros((1,),dtype=self._tstype)
            fiducials_sensor_data.sensor_data.ts[0]['nanoseconds'] = int(nanosec%1e9)
            fiducials_sensor_data.sensor_data.ts[0]['seconds'] = int(nanosec/1e9)
            fiducials_sensor_data.fiducials = fiducials

            self.fiducials_sensor_data.AsyncSendPacket(fiducials_sensor_data, lambda: None)
            self.current_fiducials_sensor_data = fiducials_sensor_data
            
            self.seqno += 1
            
            if self.SHOW_VIZ:
                cv2.imshow("Image", cv2.resize(image_viz, (450,450)))
                cv2.waitKey(1)
            # 
            time.sleep(1/self.rate)
        
    def capture_fiducials(self):

        if self.current_fiducials_sensor_data is not None:
            return self.current_fiducials_sensor_data.fiducials
        else:
            return self._fiducials()

def main():

    # not yet know what this do
    rr_args = ["--robotraconteur-jumbo-message=true"] + sys.argv
    RRC.RegisterStdRobDefServiceTypes(RRN)

    with RR.ServerNodeSetup("com.robotraconteur.fiducial.FiducialSensor",2356,argv=rr_args):
        tagdetector_obj = ArucoDetector()

        service_ctx = RRN.RegisterService("aruco_detector","com.robotraconteur.fiducial.FiducialSensor",tagdetector_obj)
        tagdetector_obj.start_detector()

        #Wait for the user to shutdown the service
        input("Server started, press enter to quit...")

        tagdetector_obj.stop_detector()

if __name__ == "__main__":
    main()