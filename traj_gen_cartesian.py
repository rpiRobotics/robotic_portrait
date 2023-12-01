import numpy as np
import matplotlib.pyplot as plt
import glob, cv2


def image2plane(img,ipad_pose,pixel2mm,pixel_paths):
    
    #find a ratio to fit image in paper
    image_center=np.array([img.shape[1]/2,img.shape[0]/2])
    cartesian_paths=[]
    for pixel_path in pixel_paths:
        cartesian_path=(pixel_path[:,:2]-image_center)*pixel2mm
        #convert pixel frame to world frame first
        cartesian_path=np.dot(np.array([[0,-1],[-1,0]]),cartesian_path.T).T
        #append 0 in z
        cartesian_path=np.hstack((cartesian_path,np.zeros((len(cartesian_path),1))))
        cartesian_path=np.dot(ipad_pose[:3,:3],cartesian_path.T).T+np.tile(ipad_pose[:3,-1],(len(cartesian_path),1))
        cartesian_paths.append(cartesian_path)
    
    return cartesian_paths

def main():
    img_name='wen_out'
    num_segments=len(glob.glob('path/pixel_path/'+img_name+'/*.csv'))
    pixel_paths=[]
    for i in range(num_segments):
        pixel_paths.append(np.loadtxt('path/pixel_path/'+img_name+'/%i.csv'%i,delimiter=',').reshape((-1,3)))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    img=cv2.imread('imgs/'+img_name+'.png')
    ipad_pose=np.loadtxt('config/ipad_pose.csv',delimiter=',')
    paper_size=np.loadtxt('config/paper_size.csv',delimiter=',')
    img_gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    pixel2mm=min(paper_size/img_gray.shape)
    
    cartesian_paths=image2plane(img, ipad_pose, pixel2mm,pixel_paths)
    for i in range(len(cartesian_paths)):

        ###plot out the path in 3D
        ax.plot(cartesian_paths[i][:,0], cartesian_paths[i][:,1], cartesian_paths[i][:,2], 'b')
        ###save path
        np.savetxt('path/cartesian_path/'+img_name+'/%i.csv'%i,cartesian_paths[i],delimiter=',')
        
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.show()
        


if __name__ == '__main__':
    main()