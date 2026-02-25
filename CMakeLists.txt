cmake_minimum_required(VERSION 3.14 FATAL_ERROR)
project(chess_rl)

# On force le standard C++17 (requis par LibTorch)
set(CMAKE_CXX_STANDARD 17)

# On demande à CMake de chercher LibTorch
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# On crée notre exécutable
add_executable(selfplay selfplay.cpp)

# On lie notre programme à LibTorch
target_link_libraries(selfplay "${TORCH_LIBRARIES}")
set_property(TARGET selfplay PROPERTY CXX_STANDARD 17)

# --- L'Arène de Tournoi ---
add_executable(arena arena.cpp)
target_link_libraries(arena "${TORCH_LIBRARIES}")
set_property(TARGET arena PROPERTY CXX_STANDARD 17)
