
#include <sndfile.h>
#include <iostream>
#include <string>
#include <vector>
#include <sstream>
#include <fstream>
#include <deque>
#include <numeric>
#include "dynamixel_sdk.h"                        // Uses Dynamixel SDK library
#include <fcntl.h>
#include <termios.h>
#include <stdlib.h>
#include <stdio.h>
#include <cmath>
#include <thread>
#include <chrono>
#include <cstring>
#include <Eigen/Dense>
#include <condition_variable>
#include <SFML/Audio.hpp>

#include <cstdlib>
#include <ctime>
#include <algorithm>
#include <random>
#include <Eigen/Dense>
#include <unsupported/Eigen/Splines>
#include <tuple>
#include "cnpy.h"
using namespace std;
using namespace Eigen;

#define AUDIO_INPUT_PATH                    "/home/taehwang/animated_robot/audio_wav/"
#define CSV_INPUT_PATH                      "/home/taehwang/animated_robot/python_function/generate_motion/csv_file/"

#define FORMAT              SF_FORMAT_WAV | SF_FORMAT_PCM_16
#define CHUNK_SIZE          1024

#define MAX_MOUTH           0 // 예시로 임의의 값 설정
#define MIN_MOUTH           300 // 예시로 임의의 값 설정

//protocol version

#define PROTOCOL_VERSION                    2.0

// Control table address
// Control table address is different in Dynamixel model

#define ADDR_PRO_TORQUE_ENABLE              64                
#define ADDR_PRO_GOAL_POSITION              116
#define ADDR_PRO_PRESENT_POSITION           132
#define ADDR_PRO_PROFILE_VELOCITY           112
#define ADDR_PRO_PRESENT_CURRENT            126
#define ADDR_PRO_FEEDFORWARD_1ST_GAIN       90
#define ADDR_PRO_POSITION_P_GAIN            84
#define ADDR_PRO_PROFILE_ACCELERATION       108
// Data Byte Length

#define LEN_PRO_GOAL_POSITION               4
#define LEN_PRO_PRESENT_POSITION            4
#define LEN_PRO_PROFILE_VELOCITY            4
#define LEN_PRO_PRESENT_CURRENT             2
#define LEN_PRO_FEEDFORWARD_1ST_GAIN        2
#define LEN_PRO_POSITION_P_GAIN             2
#define  LEN_PRO_PROFILE_ACCELERATION       4
//torque, velocity, id, baudrate, port
#define TORQUE_ENABLE                       1
#define TORQUE_DISABLE                      0

#define DXL_PROFILE_VELOCITY_HOMING         500
#define DXL_PROFILE_VELOCITY                40 
#define DXL_PROFILE_VELOCITY_CONFIGCHANGE   70
#define DXL_PROFILE_ACCELERATION 0 // 적절한 가속도 값 설정

#define DXL1_ID                             1 
#define DXL2_ID                             2
#define DXL3_ID                             3 
#define DXL4_ID                             4 
#define DXL5_ID                             5
#define DXL_NUM                             5

#define BAUDRATE                            57600 

#define DEVICENAME                          "/dev/ttyUSB0"  //sudo chmod a+rw /dev/ttyUSB0      //ls /dev/ttyUSB*
#define AUDIO_DEVICE                        "hw:1"

//DXL initial goal position
#define DEFAULT_PITCH                       1500
#define DEFAULT_ROLL_R                      1100
#define DEFAULT_ROLL_L                      2300
#define DEFAULT_YAW                         3300
#define DEFAULT_MOUTH                       2000

//robot parameter
#define PULLY_DIAMETER                      50 
#define ROBOT_HEIGHT                        100             // 베이스부터 실이 연결된 레이어 까지의 높이 small -> 100,Large -> 180
#define ROBOT_HOLE_RADIUS                   25              // 로봇 머리 구멍 반지름 small -> 25, Large -> 50
#define ROBOT_YAW_GEAR_RATIO                2               // yaw 모터가 direct하게 머리를 회전시킨다면 1로 설정 아니면 2
#define ROBOT_MOUTH_TUNE                    90              // 최대 mouse movement size in DXL dimension -> 최초값에서 입모터 조정해보면서 결정
#define ROBOT_MOUTH_BACK_COMPENSATION       1.2             // 입 움직임에 대한 뒷쪽 보상 -> TRO 논문 참조  small -> 1.2, Large -> 1.5
#define ROBOT_MOUTH_PITCH_COMPENSATION      1.5             // 입 움직임에 따른 pitch 보상

//linux

#define STDIN_FILENO                        0

//MATH
#define COMPENSATION_MAX                    0.6
#define INCLINATION                         2               //볼륨조절함수에 쓸 기울기 
#define PI                                  3.14

//stream
#define INFO_STREAM( stream ) 
//std::cout << stream <<std::endl


//파일 경로 생성
void create_file_path(std::string &file_path, const char *filename, const char *input_path);

void save_to_csv_float(const std::string& base_filename, const std::vector<float>& data, int frame_count);

void save_one_float_to_csv(const std::string& filename, float value);

void save_audio_file(const std::string& filename, const float* audiodata, sf_count_t frames, int samplerate, int channels);

std::pair<float, size_t> find_peak(const std::vector<float>& audio_buffer);

float calculate_mouth(float up2mouth, float max_MOUTH, float min_MOUTH);

void save_audio_segment(const std::string& outputFilePath, const std::vector<float>& audioData, size_t dataSize);

std::vector<float> divide_channel(const std::vector<float>& audio_data, int channels, int frames);

float moving_average(const std::deque<float>& window);

float update_final_result(std::deque<float>& moving_average_window, size_t window_size, float new_sample);

float scale_max_sample(float max_sample);

float volume_control(std::deque<float>& recent_samples, size_t window_size, float new_sample);

float scaled_result_with_moving_average(std::deque<float>& recent_samples, size_t window_size, float new_sample);

std::vector<int> RPY2DXL(double roll_f, double pitch_f, double yaw_f, double mouth_f, int mode);

bool moveDXLtoDesiredPosition(dynamixel::GroupSyncWrite& groupSyncWriteVelocity, dynamixel::GroupSyncWrite& groupSyncWritePosition, int DXL_ID[], int goal_position[], int velocity);

void update_DXL_goal_position(int DXL_goal_position[], int DXL_1, int DXL_2, int DXL_3, int DXL_4, int DXL_5);

//int를 32bit bin로 변경 (little endian)
void trans_int2bin_4(uint8_t param_goal_position[4], int dxl_goal_position);

int enable_torque(dynamixel::PacketHandler *packetHandler, dynamixel::PortHandler *portHandler, int *DXL_ID, uint8_t dxl_error);

int assignClassWith1DMiddleBoundary(double x, const vector<double>& boundaries);

double calculateRMS(const vector<float>& data, size_t start, size_t frame_length) ;

double getSegmentAverageGrad(const vector<float>& data, const string& delta = "one2one", const string& mode = "abs");

vector<vector<double>> getNextSegment_SegSeg(const vector<double>& PrevEndOneBefore, const vector<double>& PrevEnd,const cnpy::NpyArray& segment,bool gradient = true, bool gotoZero = true);

vector<vector<double>> multExpToSegment(const vector<float>& ex_energy,vector<vector<double>> ex_segment,float threshold,float div);

Eigen::VectorXd toEigenVector(const vector<double>& stdVec);

vector<vector<double>> connectTwoSegments(const vector<double>& PrevEndOneBefore,const std::array<double, 3>& lastValues, const vector<vector<double>>& nextSegment, int n_interpolate = 3);

float AM_fun(float min_open, float B, float r_k, float r_k_1, float r_k_2, float lim_delta_r);

std::tuple<float, float, float> lin_fit_fun2(float S, float X_pre, float grad_up_pre, float grad_down_pre, float del_grad, float dt);

