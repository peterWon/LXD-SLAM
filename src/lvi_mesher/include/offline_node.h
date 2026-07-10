#ifndef __LIOMESHER_NODE_H__
#define __LIOMESHER_NODE_H__

#include "glog/logging.h"
#include "ros/ros.h"
#include "utility/parameter.h"
#include "liomesher.h"
#include "posegraph.h"
#include <condition_variable>
#include <livox_ros_driver/CustomMsg.h>
#include "base_node.h"

class OfflineNode: public BaseNode{
public:
    OfflineNode(ros::NodeHandle& nh);
    ~OfflineNode();
    
    bool frontend_finished();
    bool backend_finished();
    void set_finish_flag(){
        mtx_flag_.lock();
        finish_add_data_ = true;
        mtx_flag_.unlock();
    }

private:
    bool job_finished_ = false;
};
#endif
