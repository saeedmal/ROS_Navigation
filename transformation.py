#!/usr/bin/python2.7
import rosbag
import numpy as np
import tf.transformations as tff
import csv
import matplotlib.pyplot as plt

def transform_tags_to_robotframe(tags):
    result = []
    for t in tags:
        camera_x, camera_y, camera_z = t.pose.pose.position.x, t.pose.pose.position.y, t.pose.pose.position.z
        robot_x, robot_y, robot_z = camera2Robot(camera_x, camera_y, camera_z)
        result.append((t.id, robot_x, robot_y, robot_z))
    return result


def camera2Robot(camera_x, camera_y, camera_z):
    robot_x = camera_z
    robot_y = -1 * camera_x
    robot_z = camera_y
    return robot_x, robot_y, robot_z

#odom to ground truth
def to_truth(*args):
    odom,truth_zero=args
    return truth_zero[0]+odom[0], truth_zero[1]+odom[1], truth_zero[2]+odom[2]

# find index in the grid
def find_index(r,grid_x,grid_y,grid_theta):
    x=r[0]
    y=r[1]
    theta=r[2]

    s_x = np.sort(np.concatenate((grid_x, np.array([x])), axis=0))
    index_x = np.where(s_x == x)[0]

    s_y = np.sort(np.concatenate((grid_y, np.array([y])), axis=0))
    index_y = np.where(s_y == y)[0]

    s_theta = np.sort(np.concatenate((grid_theta, np.array([theta])), axis=0))
    index_theta = np.where(s_theta == theta)[0]

    return np.array([[np.int(index_x[0])],[np.int(index_y[0])],[np.int(index_theta[0])]]).T

#filter the data at each timestamp
def bayes_filter(d,*args):

    global nx, ny, ntheta, covx, covy, covt,index_all, grid_x,grid_y,grid_theta, threshold
    bel,i=args
    bel_filtered=np.zeros((nx + 1, ny + 1, ntheta + 1))

    if d=='motion':

        mean =np.array([ index_all[i+1,0],index_all[i+1,1],index_all[i+1,2] ]) # mean of the gaussian for motion noise
        cov = [[covx, 0, 0], [0, covy, 0], [0, 0, covt]]  # covariance of gaussian for motion noise
        noise=discrete_gaussian(mean,cov)

        for ii in range(0,bel.shape[0]):
            for jj in range(0, bel.shape[1]):
                for kk in range(0, bel.shape[2]):
                    if bel[ii][jj][kk]>threshold:
                        bel_filtered += bel[ii][jj][kk] * noise

        bel_filtered/=sum(sum(sum(bel_filtered))) #normalize

    if d=='measurement':
        x=0

    return bel_filtered


def discrete_gaussian(center,cov):
    global nx, ny, ntheta
    G=np.zeros((nx + 1, ny + 1, ntheta + 1))
    mean = center
    sigma_inv=np.linalg.inv(cov)
    c=(2 * np.pi) ** -1.5 * (np.linalg.det(cov)) ** -.5
    for i in range(0,G.shape[0]):
        for j in range(0, G.shape[1]):
            for k in range(0, G.shape[2]):
                rv=np.array([i,j,k])
                a = np.dot((rv - mean).T, np.dot(sigma_inv, (rv - mean)))
                value =  c* np.exp(-0.5 * a)
                G[i][j][k]=value
    G /= sum(sum(sum(G)))
    return G

def main():

    global nx, ny, ntheta,covx, covy, covt, index_all, grid_x,grid_y,grid_theta, threshold
    # parameters of the grid
    nx = 100
    ny = 115
    ntheta = 36

    covx = .5  # covx=0.5 x_cell+...
    covy = .5  # covy=0.5 y_cell+...
    covt = .5  # 5.1 degree for theta covariance

    threshold=1e-3

    tag_data=np.array([[0],[0],[0],[0]]).T
    with open("tag_groundtruth.csv") as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            #tag_data= ndarray each row: [id,x,y,z]
            tag_data=np.concatenate((tag_data,np.array([[int(row[0])],[float(row[1])],[float(row[2])],[float(row[3])]]).T),axis=0)
        tag_data=np.delete(tag_data,0,0)

    with rosbag.Bag('lab4.bag') as bag:

        #making grid for ground truth
        grid_x = np.linspace(0, 10.0, num=nx+1)  # 10 cm for x between 0,7.68
        grid_y = np.linspace(-3, 8.5, num=ny+1)  # 10cm for y between -2.65, 0.47
        grid_theta = np.linspace(-2*np.pi, 3*np.pi, num=ntheta+1)  # 10 degrees for theta -3.14 and 3.139
        # #ground truth at t=0
        x_truth_zero=1.65
        y_truth_zero=0.15
        theta_truth_zero =0.5*np.pi
        odom_save=np.array([[0],[0],[0]]).T
        truth_save=np.array([[0],[0],[0]]).T
        index_all = np.array([[0], [0], [0]]).T

        #reading data from /odom topic
        for topic1 ,msg1, t1 in bag.read_messages(['/odom', '/tag_detections']):

            if topic1 == '/odom':
                orientation_q = msg1.pose.pose.orientation
                orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
                (roll, pitch, yaw) = tff.euler_from_quaternion(orientation_list)
                odom_save = np.concatenate(
                    (odom_save, np.array([[msg1.pose.pose.position.x], [msg1.pose.pose.position.y], [yaw]]).T), axis=0)
                pose_truth = np.array(
                    [[msg1.pose.pose.position.x + x_truth_zero], [msg1.pose.pose.position.y + y_truth_zero],
                     [yaw + theta_truth_zero]]).T
                truth_save = np.concatenate((truth_save, pose_truth), axis=0)
                ind = find_index(np.array([pose_truth[0, 0], pose_truth[0, 1], pose_truth[0, 2]]), grid_x, grid_y,
                                 grid_theta)
                index_all = np.concatenate((index_all, ind), axis=0)

            sensor_data = []
            if topic1 == '/tag_detections':
                tags = transform_tags_to_robotframe(msg1.detections)
                for tag in tags:
                    sensor_data.append(tag)
                    print(tag)


        odom_save=np.delete(odom_save,0,0)
        truth_save = np.delete(truth_save, 0, 0)
        index_all = np.delete(index_all, 0, 0)

        #subsampling from the original odom data
        t=np.zeros((423,3))
        for ii in range(0,423):
            t[ii,:]=index_all[8*ii,:]
        index_all=t


        #initial belief
        bel=np.zeros((nx+1,ny+1,ntheta+1))
        bel[17][32][18]=1 # corresponding to 1.6,0.15,0.5*pi

        loc_save=np.array([[0],[0],[0]]).T
        bel_updated=bel #initial value
        #motion step
        for i in range(0,index_all.shape[0]-1):

            args=(bel_updated,i)
            bel_updated=bayes_filter('motion',*args)
            i1,j1,k1=np.where( bel_updated==np.max(bel_updated) )

            loc=np.array([ [np.int(i1)], [np.int(j1)], [np.int(k1)] ]).T
            loc_save=np.concatenate((loc_save,loc),axis=0)


        loc_save=np.delete(loc_save,0,0)

        # print(loc_save)
        plt.figure
        plt.plot(loc_save[:,0],loc_save[:,1])
        plt.show()


if __name__ == '__main__':
    main()