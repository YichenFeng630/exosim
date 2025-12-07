#include "exoforge_core/core_node.hpp"
int main(int argc,char** argv){ rclcpp::init(argc,argv); auto n=std::make_shared<exoforge_core::CoreNode>(); rclcpp::spin(n); rclcpp::shutdown(); return 0; }
