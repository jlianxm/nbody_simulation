cmake_minimum_required(VERSION 3.2)
project(csc4005_imgui)

set(CMAKE_CXX_STANDARD 17)

find_package(SDL2 REQUIRED)
find_package(Freetype REQUIRED)
find_package(MPI REQUIRED)
find_package(Threads REQUIRED)
find_package(OpenMP REQUIRED)

set (CMAKE_CXX_STANDARD 20)
set (OpenGL_GL_PREFERENCE "GLVND")
set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS} -fopenmp")
set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")

find_package(OpenGL REQUIRED)

include_directories(
        include
        imgui
        imgui/backends
        ${SDL2_INCLUDE_DIRS}
        ${FREETYPE_INCLUDE_DIRS}
        ${MPI_CXX_INCLUDE_DIRS})

file(GLOB IMGUI_SRC
        imgui/*.cpp
        imgui/backends/imgui_impl_sdl.cpp
        imgui/backends/imgui_impl_opengl2.cpp
        imgui/misc/freetype/imgui_freetype.cpp
        imgui/misc/cpp/imgui_stdlib.cpp
        )

add_library(core STATIC ${IMGUI_SRC})
file(GLOB CSC4005_PROJECT_SRC src/graphic.cpp)

add_executable(csc4005_imgui_serial src/main.cpp ${CSC4005_PROJECT_SRC})
add_executable(csc4005_imgui_mpi src/mpi.cpp ${CSC4005_PROJECT_SRC})
add_executable(csc4005_imgui_pthread src/pthread.cpp ${CSC4005_PROJECT_SRC})
add_executable(csc4005_imgui_omp src/omp.cpp ${CSC4005_PROJECT_SRC})

get_filename_component(FONT_PATH imgui/misc/fonts/DroidSans.ttf ABSOLUTE)

target_link_libraries(core PUBLIC
        Freetype::Freetype SDL2::SDL2 OpenGL::GL ${CMAKE_DL_LIBS} Threads::Threads ${MPI_CXX_LIBRARIES})

target_link_libraries(csc4005_imgui_serial core)
target_compile_options(csc4005_imgui_serial PRIVATE -Werror -Wall -Wextra -Wpedantic)
target_link_libraries(csc4005_imgui_mpi core)
target_compile_options(csc4005_imgui_mpi PRIVATE -Werror -Wall -Wextra -Wpedantic)
target_link_libraries(csc4005_imgui_pthread core)
target_compile_options(csc4005_imgui_pthread PRIVATE -Werror -Wall -Wextra -Wpedantic)
target_link_libraries(csc4005_imgui_omp core)
target_compile_options(csc4005_imgui_omp PRIVATE -Werror -Wall -Wextra -Wpedantic)

target_compile_definitions(core PUBLIC -DImDrawIdx=unsigned)

target_compile_definitions(csc4005_imgui_serial PRIVATE -DFONT_PATH=\"${FONT_PATH}\")
target_compile_definitions(csc4005_imgui_mpi PRIVATE -DFONT_PATH=\"${FONT_PATH}\")
target_compile_definitions(csc4005_imgui_pthread PRIVATE -DFONT_PATH=\"${FONT_PATH}\")
target_compile_definitions(csc4005_imgui_omp PRIVATE -DFONT_PATH=\"${FONT_PATH}\")

if (WIN32)
    target_link_options(csc4005_imgui_serial BEFORE PRIVATE -lmingw32 -lSDL2main -lSDL2 -mwindows)
    target_link_options(csc4005_imgui_mpi BEFORE PRIVATE -lmingw32 -lSDL2main -lSDL2 -mwindows)
    target_link_options(csc4005_imgui_pthread BEFORE PRIVATE -lmingw32 -lSDL2main -lSDL2 -mwindows)
    target_link_options(csc4005_imgui_omp BEFORE PRIVATE -lmingw32 -lSDL2main -lSDL2 -mwindows)
endif()
