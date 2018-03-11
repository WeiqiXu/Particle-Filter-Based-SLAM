# -*- coding: utf-8 -*-
from scipy import io
import numpy as np
import matplotlib.pyplot as plt
import sys
import time 
import math

def get_joint(file_name):
	key_names_joint = ['acc', 'ts', 'rpy', 'gyro', 'pos', 'ft_l', 'ft_r', 'head_angles']
	data = io.loadmat(file_name+".mat")
	joint = {kn: data[kn] for kn in key_names_joint}
	return joint


def get_joint_index(joint):
    jointNames = ['Neck','Head','ShoulderL', 'ArmUpperL', 'LeftShoulderYaw','ArmLowerL','LeftWristYaw','LeftWristRoll','LeftWristYaw2','PelvYL','PelvL','LegUpperL','LegLowerL','AnkleL','FootL','PelvYR','PelvR','LegUpperR','LegLowerR','AnkleR','FootR','ShoulderR', 'ArmUpperR', 'RightShoulderYaw','ArmLowerR','RightWristYaw','RightWristRoll','RightWristYaw2','TorsoPitch','TorsoYaw','l_wrist_grip1','l_wrist_grip2','l_wrist_grip3','r_wrist_grip1','r_wrist_grip2','r_wrist_grip3','ChestLidarPan']
    joint_idx = 1
    for (i,jnames) in enumerate(joint):
        if jnames in jointNames:
            joint_idx = i
            break
    return joint_idx


def get_lidar(file_name):
	data = io.loadmat(file_name+".mat")
	lidar = []
	for m in data['lidar'][0]:
		tmp = {}
		tmp['t']= m[0][0][0]
		nn = len(m[0][0])
		if (nn != 5) and (nn != 6):			
			raise ValueError("different length!")
		tmp['pose'] = m[0][0][nn-4]
		tmp['res'] = m[0][0][nn-3]
		tmp['rpy'] = m[0][0][nn-2]
		tmp['scan'] = m[0][0][nn-1]
		
		lidar.append(tmp)
	return lidar


def bresenham2D(sx, sy, ex, ey):
  sx = int(round(sx))
  sy = int(round(sy))
  ex = int(round(ex))
  ey = int(round(ey))
  dx = abs(ex-sx)
  dy = abs(ey-sy)
  steep = abs(dy)>abs(dx)
  if steep:
    dx,dy = dy,dx # swap 

  if dy == 0:
    q = np.zeros((dx+1,1))
  else:
    q = np.append(0,np.greater_equal(np.diff(np.mod(np.arange( np.floor(dx/2), -dy*dx+np.floor(dx/2)-1,-dy),dx)),0))
  if steep:
    if sy <= ey:
      y = np.arange(sy,ey+1)
    else:
      y = np.arange(sy,ey-1,-1)
    if sx <= ex:
      x = sx + np.cumsum(q)
    else:
      x = sx - np.cumsum(q)
  else:
    if sx <= ex:
      x = np.arange(sx,ex+1)
    else:
      x = np.arange(sx,ex-1,-1)
    if sy <= ey:
      y = sy + np.cumsum(q)
    else:
      y = sy - np.cumsum(q)
  return np.vstack((x,y))


#find the closest ts to t_lidar in joint 
def closest_ts(t1,t2):
    distance=10
    ts_index=0
    for i in range(len(t2)):
        if distance>abs(t2[0][i]-t1):
            distance=abs(t2[0][i]-t1)
            ts_index=i
    return ts_index


#define matrix T_wl
def get_T_wl(lscan,pose):
    t_lidar=lidar0[lscan]['t'] #lscan---the laser scan
    t_joint=closest_ts(t_lidar,joint0['ts'])
    
    pose = np.reshape(pose, (3,1))
    
    neck_angle=joint0['head_angles'][0][t_joint]
    head_angle=joint0['head_angles'][1][t_joint]
    
    T_bl=np.zeros([4,4])
    T_bl1=  [[math.cos(neck_angle),-math.sin(neck_angle),0,0],
             [math.sin(neck_angle), math.cos(neck_angle),0,0],
             [0,0,1,0.93+0.33],
             [0,0,0,1]]
    
    T_bl2=[[math.cos(head_angle), 0,-math.sin(head_angle),0],
            [0,1,0,0],
            [math.sin(head_angle),0,math.cos(head_angle), 0],
            [0,0,0,1]]
    T_bl3=[[1,0,0,0],
           [0,1,0,0],
           [0,0,1,0.15],
           [0,0,0,1]]
    T_bl23=np.dot(T_bl2,T_bl3)
    T_bl=np.dot(T_bl1,T_bl23)
    T_wb=[[  math.cos(pose[2]),-math.sin(pose[2]),0,pose[0]],
             [math.sin(pose[2]),math.cos(pose[2]),0,pose[1]],
             [0,0,1,0],
             [0,0,0,1]]
    T_wl=np.dot(T_wb,T_bl)
    return T_wl



# convert Lidar data to the world frame
def get_L2W(lscan,T_wl):
    radius=lidar0[lscan]['scan']
    radius=radius[0,:].tolist()
    i=0
    
    theta = np.arange(-135,135.25,0.25)*np.pi/float(180)
    theta=theta.tolist()

    while i<len(radius):
        if radius[i]>10 or radius[i]<0.2: #delete any element that is larger than 10m or smaller than 0.2m
            del radius[i]
            del theta[i]
        else:
            i=i+1
          
    lidar_frame=np.zeros([4,len(radius[:])])
    lidar_world=np.zeros([3,len(radius[:])])
    for i in range(len(radius)):
        lidar_frame[:,i]=np.transpose([radius[i]*math.cos(theta[i]),radius[i]*math.sin(theta[i]),0,1])
        lidar_world[:,i]=np.dot(T_wl,lidar_frame[:,i])[0:3]

    return lidar_world




def update_occupancy(lidar_world,occupancy,pose):
    n=len(lidar_world[0][:])
    
    occupancy = np.array(occupancy)
    pose = np.reshape(pose,(3,1)) 

    
    rob_col = int(np.round(pose[0]/0.1+occupancy.shape[1]/2)-1 )
    rob_row = int(np.round(pose[1]/0.1+occupancy.shape[0]/2)-1 )
    

    
    for i in range(n):
        
        #if lidar_world[2,i]<0.2: # z>0.2
        #    continue
        
        column=int(round(lidar_world[0,i]/0.1+occupancy.shape[1]/2)-1 )
        row=int(round(lidar_world[1,i]/0.1+occupancy.shape[0]/2)-1)
        
        if occupancy[row][column]<10*math.log(8):
            occupancy[row][column]=occupancy[row][column]+math.log(8) 
            
        if(rob_row < row) :   
            track=bresenham2D(rob_row,rob_col,row,column)
        else:
            track=bresenham2D(row, column, rob_row,rob_col)

        for j in range(1, track.shape[1]-1):
            r = int(track[0,j])
            c = int(track[1,j])
            if occupancy[r][c]>-10*math.log(8):
                occupancy[r][c] = occupancy[r][c]-math.log(8)
            
    return occupancy      

joint0=get_joint("/Users/xuweiqi/Desktop/ECE276A Project3/trainset/joint/train_joint0")
lidar0=get_lidar("/Users/xuweiqi/Desktop/ECE276A Project3/trainset/lidar/train_lidar0")
occupancy=np.zeros((300, 300))
N=len(lidar0)
 

'''

for i in range(0,N,50):
    pose=lidar0[i]['pose']
    T_wl=get_T_wl(i,pose)
#   print("Frame = {}".format(i))
#   print("Pose  = {}".format(pose))
#   print("T_wl  = {}".format(T_wl))
    lidar_world=get_L2W(i,T_wl)
#plt.plot(lidar_world[0,:],lidar_world[1,:],'o') 
    update_occupancy(lidar_world,occupancy,pose)
#plt.imshow(occupancy)  
    
'''

######  particle filter
# initialize particles
startPoint = 0

particlePose=np.zeros([100,3])
NewPose=np.zeros([100,3])
alpha=0.01*np.ones([100,1])
pose=np.zeros([1,3])
T_wl=get_T_wl(startPoint,pose)
lidar_world=get_L2W(startPoint,T_wl)
occupancy = update_occupancy(lidar_world,occupancy,pose)

plt.imshow(occupancy)
plt.pause(0.001)



def PredictStep(particlePose,ChangeinPose):
    for i in range(100):
        NewPose[i,:]=particlePose[i,:]+ChangeinPose
    for i in range(100):
        NewPose[i,:]=NewPose[i,:]+np.transpose(np.array([np.random.normal(0,0.03,1),np.random.normal(0,0.03,1),np.random.normal(0,0.01,1)]))
    
    return NewPose


def UpdateStep(particlePose,alpha,occupancy_old,lscan):
    alphaOut=np.zeros(alpha.shape)
    corrValues = np.zeros(alpha.shape)
    
    for i in range(100):
        
        T_wl = get_T_wl(lscan,particlePose[i,:])
        lidar_world = get_L2W(lscan,T_wl)
        lidarOnlyOccupancy = update_occupancy(lidar_world,np.zeros(occupancy_old.shape),particlePose[i,:])
        
        lidarOnlyOccupancy[lidarOnlyOccupancy > 0] = 1
        lidarOnlyOccupancy[lidarOnlyOccupancy <= 0] = 0
        
        occupancy_old[occupancy_old > 0] = 1
        occupancy_old[occupancy_old <= 0] = -5
        
        corrMap = lidarOnlyOccupancy == occupancy_old
        corrValues[i] = np.sum(corrMap.flatten())
        
    print("Max Corr Value: {}".format(np.amax(corrValues)))
    alphaOut = corrValues-np.amax(corrValues)
        
    for i in range(0,len(alphaOut)):
        alphaOut[i] = alpha[i]*math.exp(alphaOut[i])

    alphaOut = alphaOut/np.sum(alphaOut)

    return particlePose, alphaOut, np.amax(corrValues)


for i in range(startPoint+1,N,50):
    print("On Frame {}".format(i))
    
    ChangeinPose=lidar0[i]['pose']-lidar0[i-1]['pose']

    maxCorrValue = 0 
    
    while maxCorrValue < 75:
        new_particlePose=PredictStep(particlePose,ChangeinPose)
    
        new_particlePose, new_alpha, maxCorrValue = UpdateStep(new_particlePose,alpha,np.copy(occupancy),i)
    
    particlePose = new_particlePose
    alpha = new_alpha
    
    bestParticleIndex = np.argmax(alpha)
    newPose = particlePose[bestParticleIndex,:]
    
    numEffectiveParticles = 0
    for j in range(0, len(alpha)):
        numEffectiveParticles = numEffectiveParticles + alpha[j]**2
    numEffectiveParticles = 1/numEffectiveParticles
    
    print("Best particle value: {}".format(newPose))
    print("Best alpha value   : {}".format(alpha[bestParticleIndex]))
    print("Effective Particles: {}".format(numEffectiveParticles))
    
    #newPose = lidar0[i]['pose']
    
    T_wl=get_T_wl(i,newPose)
    lidar_world=get_L2W(i,T_wl)
    
    occupancy = update_occupancy(lidar_world, occupancy, newPose)
    
    plt.imshow(occupancy)
    plt.pause(0.0001)
    
    if numEffectiveParticles < 5:
        print("Resampling")
        for j in range(0, len(alpha)):
            particlePose[j,:] = newPose
            alpha[j] = 1/100
        
#plt.imshow(occupancy)

