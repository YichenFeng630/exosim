#pragma once
#include <rclcpp/rclcpp.hpp>
namespace exoforge_interfaces { class InterfacesNode : public rclcpp::Node { public: InterfacesNode() : Node("interfaces_node") {} }; }
