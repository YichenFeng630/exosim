#include "exoforge_bringup/bringup_node.hpp"
int main(int argc,char** argv){ rclcpp::init(argc,argv); auto n=std::make_shared<exoforge_bringup::BringupNode>(); rclcpp::spin(n); rclcpp::shutdown(); return 0; }
