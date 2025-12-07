#pragma once
#include <rclcpp/rclcpp.hpp>
namespace exoforge_safety { class SafetyNode : public rclcpp::Node { public: SafetyNode() : Node("safety_node") {} }; }
