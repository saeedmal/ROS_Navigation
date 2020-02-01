#include <vector>
#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <apriltags_ros/AprilTagDetection.h>
#include <apriltags_ros/AprilTagDetectionArray.h>

#include <boost/foreach.hpp>
#define foreach BOOST_FOREACH

using namespace std;
using namespace apriltags_ros;

vector<double> camera2Robot(double camera_x, double camera_y, double camera_z) {
    double robot_x = camera_z;
    double robot_y = -1 * camera_z;
    double robot_z = camera_y;

    vector<double> result;
    result.push_back(robot_x);
    result.push_back(robot_y);
    result.push_back(robot_z);

    return result;
}

string toString(int id) {
    stringstream ss;
    ss << id;
    return ss.str();
}

map< string, vector<double> > transformTagsToRobotFrame(AprilTagDetectionArray::ConstPtr arr) {
    map< string, vector<double> > result;
    foreach(AprilTagDetection const detection, arr->detections) {
        vector<double> robot_point = camera2Robot( detection.pose.pose.position.x
                                              , detection.pose.pose.position.y
                                              , detection.pose.pose.position.z);
        result[toString(detection.id)] = robot_point;
    }
    return result;
}

int main() {
    rosbag::Bag bag;
    bag.open("lab4.bag", rosbag::bagmode::Read);

    std::vector<std::string> topics;
    topics.push_back(std::string("/tag_detections"));

    rosbag::View view(bag, rosbag::TopicQuery(topics));

    foreach(rosbag::MessageInstance const msg, view)
    {
        AprilTagDetectionArray::ConstPtr arr = msg.instantiate<apriltags_ros::AprilTagDetectionArray>();
        if(arr != NULL) {
            map< string, vector<string> > result = transformTagsToRobotFrame(arr);
        }
    }
}
