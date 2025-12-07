#include "exoforge_therapy/therapy_node.hpp"
int main(int argc,char** argv){ rclcpp::init(argc,argv); auto n=std::make_shared<exoforge_therapy::TherapyNode>(); rclcpp::spin(n); rclcpp::shutdown(); return 0; }
