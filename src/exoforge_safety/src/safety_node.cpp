#include "exoforge_safety/safety_node.hpp"
int main(int argc,char** argv){ rclcpp::init(argc,argv); auto n=std::make_shared<exoforge_safety::SafetyNode>(); rclcpp::spin(n); rclcpp::shutdown(); return 0; }
