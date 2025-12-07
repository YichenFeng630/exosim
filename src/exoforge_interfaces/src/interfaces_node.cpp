#include "exoforge_interfaces/interfaces_node.hpp"
int main(int argc,char** argv){ rclcpp::init(argc,argv); auto n=std::make_shared<exoforge_interfaces::InterfacesNode>(); rclcpp::spin(n); rclcpp::shutdown(); return 0; }
