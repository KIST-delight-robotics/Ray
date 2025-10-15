# CMake 최소 요구 버전
cmake_minimum_required(VERSION 3.15)

# 프로젝트 이름 및 언어 설정
project(Ray LANGUAGES CXX)

# C++ 표준 설정 (C++17 이상 권장)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# ==================================================================
# 의존성 관리 (FetchContent)
include(FetchContent)

# 1. nlohmann-json (헤더 전용 라이브러리)
# JSON 파싱에 사용
FetchContent_Declare(
  json
  GIT_REPOSITORY https://github.com/nlohmann/json.git
  GIT_TAG        v3.11.3 # 안정적인 버전으로 고정
)

# 2. cnpy (Numpy .npy/.npz 파일 로더)
# Numpy .npy/.npz 파일 로딩에 사용
FetchContent_Declare(
  cnpy
  GIT_REPOSITORY https://github.com/rogersce/cnpy.git
)

# 3. Eigen (헤더 전용 선형대수 라이브러리)
# 기구학 계산에 사용
FetchContent_Declare(
  Eigen3
  GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
  GIT_TAG        3.4.0 # 특정 버전 고정
)

# 4. SFML (오디오 및 그래픽 라이브러리)
# 오디오 처리 및 재생에 사용
set(SFML_BUILD_GRAPHICS OFF)
set(SFML_BUILD_WINDOW OFF)
set(SFML_BUILD_NETWORK OFF)
FetchContent_Declare(
  SFML
  GIT_REPOSITORY https://github.com/SFML/SFML.git
  GIT_TAG        2.6.1 # 특정 버전 고정
)

# 5. ixwebsocket (웹소켓 통신)
FetchContent_Declare(
  ixwebsocket
  GIT_REPOSITORY https://github.com/machinezone/IXWebSocket.git
  CMAKE_ARGS -DUSE_WS=OFF # 웹소켓 클라이언트 기능만 사용 (서버 불필요)
             -DUSE_TLS=OFF # TLS(OpenSSL) 지원 비활성화 (필요 시 ON)
)

# 선언된 모든 의존성을 다운로드하고 빌드하여 사용 가능하게 만듭니다.
FetchContent_MakeAvailable(json cnpy Eigen3 SFML ixwebsocket)

# ==================================================================
# 프로젝트 소스 코드 및 실행 파일 정의
# ==================================================================

# 실행 파일 생성 위치 설정
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR})

# 실행 파일 생성
add_executable(Ray
    cpp/main.cpp
    cpp/Macro_function.cpp
    # 여기에 다른 .cpp 파일들을 추가하세요.
    # 예: cpp/audio_processing.cpp
    #    cpp/robot_control.cpp
)

# 헤더 파일 검색 경로 추가
target_include_directories(Ray PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/cpp"
    ${cnpy_SOURCE_DIR}
    third_party/DynamixelSDK-3.7.31/c++/include/dynamixel_sdk
)

target_link_directories(Ray PRIVATE
    third_party/DynamixelSDK-3.7.31/c++/build/linux64
)

# 실행 파일에 라이브러리 링크
target_link_libraries(Ray PRIVATE
    nlohmann_json::nlohmann_json
    cnpy
    Eigen3::Eigen
    sfml-audio
    dxl_x64_cpp
    ixwebsocket
    sndfile
)
