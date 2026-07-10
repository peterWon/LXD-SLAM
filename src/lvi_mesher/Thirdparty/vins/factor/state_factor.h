#ifndef __STATE_FACTOR__
#define __STATE_FACTOR__

#include <eigen3/Eigen/Dense>
#include <ceres/ceres.h>

namespace cvins{
  class NodeStateCostFunction{
     public:
    static ceres::CostFunction* CreateAutoDiffCostFunction(
        const Eigen::Vector3d& P, 
        const Eigen::Matrix3d& R,
        const Eigen::Vector3d& V, 
        const Eigen::Vector3d& Ba, 
        const Eigen::Vector3d& Bg,
        const std::vector<double>& weights) {
      return new ceres::AutoDiffCostFunction<
          NodeStateCostFunction, 
          ceres::DYNAMIC, /* residuals */
          7, /* translation and rotation variables */
          9 /* velocity and bias variables */>(
            new NodeStateCostFunction(P, R, V, Ba, Bg, weights), 15);
    }
    
    template <typename T>
    bool operator()(const T* const pose, 
                    const T* const speed,
                    T* const residual) const {
      size_t offset = 0;
      Eigen::Matrix<T,3,1> pos(pose[0], pose[1], pose[2]);
      Eigen::Map<Eigen::Matrix<T,3,1>> rp(residual+offset);
      rp = T(weights_[0]) * (pos - P_.cast<T>()); 
      
      offset += 3; 
      Eigen::Quaternion<T> q(pose[6], pose[3], pose[4], pose[5]);
      q.normalize();
      Eigen::Quaternion<T> prior_rotation_ = Eigen::Quaterniond(R_).cast<T>();
      prior_rotation_.normalize();
      Eigen::Quaternion<T> prior_quaternion_inverse = prior_rotation_.conjugate();
      Eigen::Quaternion<T> relative_quaternion = prior_quaternion_inverse * q;
      Eigen::Matrix<T, 3, 1> rotation_residual = T(2.0) * relative_quaternion.vec();
      Eigen::Map<Eigen::Matrix<T,3,1>> rq(residual+offset);
      rq = T(weights_[1]) * rotation_residual;

      offset += 3;
      Eigen::Matrix<T,3,1> vel(speed[0], speed[1], speed[2]);
      Eigen::Map<Eigen::Matrix<T,3,1>> rv(residual+offset);
      rv = T(weights_[2]) * (vel - V_.cast<T>());
      
      offset += 3;
      Eigen::Matrix<T,3,1> bias_acc(speed[3], speed[4], speed[5]);
      Eigen::Map<Eigen::Matrix<T,3,1>> rba(residual+offset);
      rba = T(weights_[3]) * (bias_acc - Ba_.cast<T>());
      offset += 3;
      
      Eigen::Matrix<T,3,1> bias_gyr(speed[6], speed[7], speed[8]);
      Eigen::Map<Eigen::Matrix<T,3,1>> rbg(residual+offset);
      rbg = T(weights_[4]) * (bias_gyr - Bg_.cast<T>());

      return true;
    }

  private:
    NodeStateCostFunction(const Eigen::Vector3d& P, 
        const Eigen::Matrix3d& R, const Eigen::Vector3d& V, 
        const Eigen::Vector3d& Ba, const Eigen::Vector3d& Bg,
        const std::vector<double>& weights)
        : P_(P), R_(R), V_(V), Ba_(Ba), Bg_(Bg), weights_(weights) {}

    NodeStateCostFunction(const NodeStateCostFunction&) = delete;
    NodeStateCostFunction& operator=(
        const NodeStateCostFunction&) = delete;

    const Eigen::Vector3d P_;
    const Eigen::Matrix3d R_;
    const Eigen::Vector3d V_;
    const Eigen::Vector3d Ba_;
    const Eigen::Vector3d Bg_;
    const std::vector<double> weights_;
  };
}

#endif//__STATE_FACTOR__