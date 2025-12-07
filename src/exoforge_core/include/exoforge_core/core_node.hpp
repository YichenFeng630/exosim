#pragma once
#include <rclcpp/rclcpp.hpp>
namespace exoforge_core { class CoreNode : public rclcpp::Node { public: CoreNode() : Node("core_node") {} }; }
