#include "exoforge_sensing/sensing_node.hpp"
int main(int argc,char** argv){ rclcpp::init(argc,argv); auto n=std::make_shared<exoforge_sensing::SensingNode>(); rclcpp::spin(n); rclcpp::shutdown(); return 0; }
