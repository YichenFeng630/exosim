#pragma once
#include <rclcpp/rclcpp.hpp>
namespace exoforge_simulation { class SimulationNode : public rclcpp::Node { public: SimulationNode() : Node("simulation_node") {} }; }
