#pragma once
#include <rclcpp/rclcpp.hpp>
namespace exoforge_therapy { class TherapyNode : public rclcpp::Node { public: TherapyNode() : Node("therapy_node") {} }; }
