cmake_minimum_required(VERSION 3.2)
project( NER )
find_package( OpenCV REQUIRED )
aux_source_directory( src/. src_list )
add_executable( NER ${src_list} )
include(mitielib/cmake)
target_link_libraries (NER ${OpenCV_LIBS})
target_link_libraries(NER mitie)

