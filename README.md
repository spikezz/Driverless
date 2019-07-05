Projekt Driverless Formula Student
====  
**bei Team Karat**
further progress of Driverless Project

    Inhalt:

    1. Imitation learning Algorithmus

    2. reinforcement learning DDPG Algorithmus

    3.Schnittstelle zwischen Unreal und ROS

    4.classic controller:predict controller+PD+PID
    
**Software feature**

1.data ploter
    
![image](https://github.com/spikezz/Driverless/blob/master/images/data_visualizer.gif)

2.camera angle

![image](https://github.com/spikezz/Driverless/blob/master/images/different_camera.gif)

3.global map

![image](https://github.com/spikezz/Driverless/blob/master/images/global_map.gif)

4.lidar vision

![image](https://github.com/spikezz/Driverless/blob/master/images/lidar_vision.gif)

<br />
<br />
<br />
<br />

**New feature updated 5.Juli.2019**

<br />
<br />
  
*Senario design*

<br />
<br />

    Full map

![image](https://github.com/spikezz/Driverless/blob/master/images/track_map.png)

<br />
<br />

    1.both side noise
    
![image](https://github.com/spikezz/Driverless/blob/master/images/both_noise.png)

<br />
<br />

    2.left noise
    
![image](https://github.com/spikezz/Driverless/blob/master/images/link_noise.png)

<br />
<br />

    3.right noise
    
![image](https://github.com/spikezz/Driverless/blob/master/images/right_noise.png)

<br />
<br />

    4.turning left with increasing curverature

![image](https://github.com/spikezz/Driverless/blob/master/images/link_c_steigen.png)

<br />
<br />

    5.turning right with increasing curverature

![image](https://github.com/spikezz/Driverless/blob/master/images/right_c_steigen.png)

<br />
<br />

    6.clip track
 
![image](https://github.com/spikezz/Driverless/blob/master/images/clip_track.png)

<br />
<br />

    7.T track
 
![image](https://github.com/spikezz/Driverless/blob/master/images/T_track.png)

<br />
<br />

    8.sad face track
 
![image](https://github.com/spikezz/Driverless/blob/master/images/three.png)

<br />
<br />

    9.noise free straight track
 
![image](https://github.com/spikezz/Driverless/blob/master/images/noise_free_straight.png)

<br />
<br />
<br />
<br />

*New function of the core*

<br />
<br />

![image](https://github.com/spikezz/Driverless/blob/master/images/new_feature.gif)

<br />
<br />

    1.Better visulization for perception
    
<br />

    2.Physic synchronization between controller and Unreal simulatuion

<br />

    3.Re-tuning the parameter of classic controller

<br />

    4.Development of Path-planning algorithm with is robust against complex senario

<br />

    5.Simulation of noise on perception result IMU sensor, speed sensor, and steering sensor

<br />

    6.Simulation of optical cover for cone
