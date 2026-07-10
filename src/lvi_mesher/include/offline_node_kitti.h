#ifndef __OFFLINE_NODE_KITTI_H__
#define __OFFLINE_NODE_KITTI_H__

#include "glog/logging.h"
#include "ros/ros.h"
#include "utility/parameter.h"
#include "liomesher.h"
#include "posegraph.h"
#include <condition_variable>
#include <livox_ros_driver/CustomMsg.h>
#include "base_node.h"

class OfflineNodeKITTI : public BaseNode{
public:
    OfflineNodeKITTI();
    ~OfflineNodeKITTI();

    bool frontend_finished();
    bool backend_finished();
    void set_finish_flag(){
        finish_add_data_ = true;
    }
private:
    bool job_finished_ = false;
};
#endif
