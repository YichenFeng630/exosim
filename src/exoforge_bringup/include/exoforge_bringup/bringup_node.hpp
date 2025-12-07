#pragma once
#include <rclcpp/rclcpp.hpp>
namespace exoforge_bringup { class BringupNode : public rclcpp::Node { public: BringupNode() : Node("bringup_node") {} }; }
