find_package(Glog REQUIRED)
include_directories(${Glog_INCLUDE_DIRS})

find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
include_directories(${Python3_INCLUDE_DIRS})

find_package(Python3 COMPONENTS NumPy)
include_directories(${Python3_NumPy_INCLUDE_DIRS})

find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIRS})

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})


set(third_party_libs
    glog gflags Python3::Python Python3::NumPy ${OpenCV_LIBS}
    )
