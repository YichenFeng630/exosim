#pragma once
#include <rclcpp/rclcpp.hpp>
namespace exoforge_sensing { class SensingNode : public rclcpp::Node { public: SensingNode() : Node("sensing_node") {} }; }
