//
// Created by bzeren on 05.10.2023.
//

#pragma once

#include <utility>

#include <rclcpp/rclcpp.hpp>

#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/camera_info.hpp>

#include <geometry_msgs/msg/transform_stamped.h>
#include <geometry_msgs/msg/detail/transform_stamped__struct.hpp>

#include <tf2_eigen/tf2_eigen.hpp>

#include <Eigen/Dense>

#include <opencv2/opencv.hpp>

#include <cv_bridge/cv_bridge.h>

enum class ImageType {
    RAW,
    RECTIFIED
};

namespace color_point_cloud {
    class CameraType {
    public:
        CameraType(std::string image_topic, std::string camera_info_topic) :
                image_topic_(std::move(image_topic)), camera_info_topic_(std::move(camera_info_topic)),
                is_info_initialized_(false), is_transform_initialized_(false), image_width_(0), image_height_(0) {

        }

        void set_cv_image(const sensor_msgs::msg::Image::ConstSharedPtr &msg, ImageType image_type) {
            cv_bridge::CvImageConstPtr cv_ptr;
            if (image_type == ImageType::RAW) {
                try {
                    cv_ptr = cv_bridge::toCvShare(msg, get_image_msg()->encoding);
                } catch (cv_bridge::Exception &e) {
//                    RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
                    return;
                }
                cv::undistort(cv_ptr->image, cv_image_, get_camera_matrix_cv(),
                              get_distortion_matrix_cv());
            } else if (image_type == ImageType::RECTIFIED) {
                try {
                    cv_ptr = cv_bridge::toCvShare(msg, get_image_msg()->encoding);
                } catch (cv_bridge::Exception &e) {
//                    RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
                    return;
                }
                cv_image_ = cv_ptr->image;
            } else {
//                RCLCPP_ERROR(this->get_logger(), "Unknown image type");
                return;
            }
        }

        void set_image_msg(const sensor_msgs::msg::Image::ConstSharedPtr &msg) {
            image_msg_ = msg;
        }

        void set_camera_info(const sensor_msgs::msg::CameraInfo::ConstSharedPtr &msg) {
            camera_info_ = msg;
        }

        cv::Mat get_cv_image() {
            return cv_image_;
        }

        sensor_msgs::msg::Image::ConstSharedPtr get_image_msg() {
            return image_msg_;
        }

        sensor_msgs::msg::CameraInfo::ConstSharedPtr get_camera_info() {
            return camera_info_;
        }

        bool is_info_initialized() const {
            return is_info_initialized_;
        }

        bool is_transform_initialized() const {
            return is_transform_initialized_;
        }

        void set_camera_utils(const sensor_msgs::msg::CameraInfo::ConstSharedPtr &msg) {
            camera_frame_id_ = msg->header.frame_id;
            distortion_model_ = msg->distortion_model;

            image_width_ = msg->width;
            image_height_ = msg->height;

            camera_matrix_(0, 0) = msg->k[0];
            camera_matrix_(0, 2) = msg->k[2];
            camera_matrix_(1, 1) = msg->k[4];
            camera_matrix_(1, 2) = msg->k[5];
            camera_matrix_(2, 2) = 1;

            camera_matrix_cv_ = (cv::Mat_<double>(3, 3)
                    << msg->k[0], 0.000000, msg->k[2], 0.000000, msg->k[4], msg->k[5], 0.000000, 0.000000, 1.000000);

            rectification_matrix_(0, 0) = msg->r[0];
            rectification_matrix_(0, 1) = msg->r[1];
            rectification_matrix_(0, 2) = msg->r[2];
            rectification_matrix_(1, 0) = msg->r[3];
            rectification_matrix_(1, 1) = msg->r[4];
            rectification_matrix_(1, 2) = msg->r[5];
            rectification_matrix_(2, 0) = msg->r[6];
            rectification_matrix_(2, 1) = msg->r[7];
            rectification_matrix_(2, 2) = msg->r[8];

            projection_matrix_(0, 0) = msg->p[0];
            projection_matrix_(0, 2) = msg->p[2];
            projection_matrix_(1, 1) = msg->p[5];
            projection_matrix_(1, 2) = msg->p[6];
            projection_matrix_(2, 2) = 1;

            distortion_matrix_(0, 0) = msg->d[0];
            distortion_matrix_(0, 1) = msg->d[1];
            distortion_matrix_(0, 2) = msg->d[3];
            distortion_matrix_(0, 3) = msg->d[4];
            distortion_matrix_(0, 4) = msg->d[2];

            distortion_matrix_cv_ = (cv::Mat_<double>(1, 5) << msg->d[0], msg->d[1], msg->d[3], msg->d[4], msg->d[2]);

            is_info_initialized_ = true;
        }

        void set_lidar_to_camera_matrix(geometry_msgs::msg::TransformStamped &msg) {
            Eigen::Affine3d affine_lidar_to_camera_matrix = tf2::transformToEigen(msg);

            lidar_to_camera_matrix_ = Eigen::Matrix4d::Identity();
            lidar_to_camera_matrix_ = affine_lidar_to_camera_matrix.matrix();

            std::cout << "lidar_to_camera_matrix_ " << lidar_to_camera_matrix_ << std::endl;

            is_transform_initialized_ = true;
        }

        Eigen::Vector2d project_lidar_to_camera(Eigen::Vector3d point_3d) {
            const auto pt_2d = (point_3d.template head<2>() / point_3d.z()).eval();
            const auto pt_d = pinhole_distort(pt_2d);

            const auto& fx = camera_matrix_(0, 0);
            const auto& fy = camera_matrix_(1, 1);
            const auto& cx = camera_matrix_(0, 2);
            const auto& cy = camera_matrix_(1, 2);

            return {fx * pt_d(0) + cx, fy * pt_d(1) + cy};
        }

        Eigen::Vector2d pinhole_distort(const Eigen::Vector2d& pt) {
            const auto& k1 = distortion_matrix_(0, 0);
            const auto& k2 = distortion_matrix_(0, 1);
            const auto& k3 = distortion_matrix_(0, 3);

            const auto& p1 = distortion_matrix_(0, 4);
            const auto& p2 = distortion_matrix_(0, 2);

            const auto x2 = pt.x() * pt.x();
            const auto y2 = pt.y() * pt.y();
            const auto xy = pt.x() * pt.y();

            const auto r2 = x2 + y2;
            const auto r4 = r2 * r2;
            const auto r6 = r2 * r4;

            const auto r_coeff = 1.0 + k1 * r2 + k2 * r4 + k3 * r6;
            const auto t_coeff1 = 2.0 * pt.x() * pt.y();
            const auto t_coeff2 = r2 + 2.0 * x2;
            const auto t_coeff3 = r2 + 2.0 * y2;

            const auto x = r_coeff * pt.x() + p1 * t_coeff1 + p2 * t_coeff2;
            const auto y = r_coeff * pt.y() + p1 * t_coeff3 + p2 * t_coeff1;

            return Eigen::Vector2d(x, y);

        }

        void set_lidar_to_camera_projection_matrix() {
            lidar_to_camera_projection_matrix_ = projection_matrix_ * lidar_to_camera_matrix_.block<4, 4>(0, 0);
            std::cout << "lidar_to_camera_projection_matrix_ " << lidar_to_camera_projection_matrix_ << std::endl;
        }

        std::string get_image_topic() {
            return image_topic_;
        }

        std::string get_camera_info_topic() {
            return camera_info_topic_;
        }

        std::string get_camera_frame_id() {
            return camera_frame_id_;
        }

        std::string get_distortion_model() {
            return distortion_model_;
        }

        double get_image_width() const {
            return image_width_;
        }

        double get_image_height() const {
            return image_height_;
        }

        Eigen::Matrix<double, 3, 3> get_camera_matrix() {
            return camera_matrix_;
        }

        cv::Mat get_camera_matrix_cv() {
            return camera_matrix_cv_;
        }

        Eigen::Matrix<double, 3, 3> get_rectification_matrix() {
            return rectification_matrix_;
        }

        Eigen::Matrix<double, 3, 4> get_projection_matrix() {
            return projection_matrix_;
        }

        Eigen::Matrix<double, 1, 5> get_distortion_matrix() {
            return distortion_matrix_;
        }

        cv::Mat get_distortion_matrix_cv() {
            return distortion_matrix_cv_;
        }

        Eigen::Matrix4d get_lidar_to_camera_matrix() {
            return lidar_to_camera_matrix_;
        }

        Eigen::Matrix<double, 3, 4> get_lidar_to_camera_projection_matrix() {
            return lidar_to_camera_projection_matrix_;
        }



    private:
        std::string image_topic_;
        std::string camera_info_topic_;

        cv::Mat cv_image_;

        sensor_msgs::msg::Image::ConstSharedPtr image_msg_;
        sensor_msgs::msg::CameraInfo::ConstSharedPtr camera_info_;

        bool is_info_initialized_;
        bool is_transform_initialized_;

        std::string camera_frame_id_;
        std::string distortion_model_;

        double image_width_;
        double image_height_;

        Eigen::Matrix<double, 3, 3> camera_matrix_;
        Eigen::Matrix<double, 3, 3> rectification_matrix_;
        Eigen::Matrix<double, 3, 4> projection_matrix_;
        Eigen::Matrix<double, 1, 5> distortion_matrix_;

        cv::Mat camera_matrix_cv_;
        cv::Mat distortion_matrix_cv_;

        Eigen::Matrix4d lidar_to_camera_matrix_;

        Eigen::Matrix<double, 3, 4> lidar_to_camera_projection_matrix_;
    };

    typedef std::shared_ptr<CameraType> CameraTypePtr;
    typedef std::shared_ptr<const CameraType> CameraTypeConstPtr;

} // namespace color_point_cloud

