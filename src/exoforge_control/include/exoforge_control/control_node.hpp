#pragma once
#include <rclcpp/rclcpp.hpp>
namespace exoforge_control { class ControlNode : public rclcpp::Node { public: ControlNode() : Node("control_node") {} }; }
