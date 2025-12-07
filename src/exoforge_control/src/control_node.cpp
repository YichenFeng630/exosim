#include "exoforge_control/control_node.hpp"
int main(int argc,char** argv){ rclcpp::init(argc,argv); auto n=std::make_shared<exoforge_control::ControlNode>(); rclcpp::spin(n); rclcpp::shutdown(); return 0; }
