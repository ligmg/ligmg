cmake_minimum_required(VERSION 3.9)

project(ligmg_examples LANGUAGES C CXX)

find_package(LIGMG REQUIRED)

add_executable(main main.cpp)
target_link_libraries(main PUBLIC ligmg::LIGMG)
