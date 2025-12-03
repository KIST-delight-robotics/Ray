// C++ Standard Library
#include <algorithm>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <tuple>
#include <vector>

// Linux/System headers
#include <fcntl.h>
#include <termios.h>

// External Libraries
#include <Eigen/Dense>
#include <unsupported/Eigen/Splines>
#include <SFML/Audio.hpp>
#include <sndfile.h>

#include "cnpy.h"
#include "dynamixel_sdk.h" // Uses Dynamixel SDK library

using namespace std;
using namespace Eigen;

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
#define ADDR_PRO_GOAL_VELOCITY              104
#define ADDR_PRO_PRESENT_POSITION           132
#define ADDR_PRO_PRESENT_VELOCITY           128
#define ADDR_PRO_PROFILE_VELOCITY           112
#define ADDR_PRO_PRESENT_CURRENT            126
#define ADDR_PRO_FEEDFORWARD_1ST_GAIN       90
#define ADDR_PRO_FEEDFORWARD_2ND_GAIN       88
#define ADDR_PRO_POSITION_P_GAIN            84
#define ADDR_PRO_PROFILE_ACCELERATION       108
#define ADDR_PRO_DRIVE_MODE                 10
#define ADDR_PRO_OPERATING_MODE             11
#define ADDR_PRO_BAUDRATE                   8
#define ADDR_PRO_RETURN_DELAY_TIME          9
#define ADDR_PRO_VELOCITY_I_GAIN            76
#define ADDR_PRO_VELOCITY_P_GAIN            78

// Data Byte Length
#define LEN_PRO_GOAL_POSITION               4
#define LEN_PRO_GOAL_VELOCITY               4
#define LEN_PRO_PRESENT_POSITION            4
#define LEN_PRO_PRESENT_VELOCITY            4
#define LEN_PRO_PROFILE_VELOCITY            4
#define LEN_PRO_PRESENT_CURRENT             2
#define LEN_PRO_FEEDFORWARD_1ST_GAIN        2
#define LEN_PRO_FEEDFORWARD_2ND_GAIN        2
#define LEN_PRO_POSITION_P_GAIN             2
#define LEN_PRO_PROFILE_ACCELERATION        4
#define LEN_PRO_VELOCITY_I_GAIN             2
#define LEN_PRO_VELOCITY_P_GAIN             2

//torque, velocity, id, baudrate, port
#define TORQUE_ENABLE                       1
#define TORQUE_DISABLE                      0

#define DXL_PROFILE_VELOCITY_HOMING         500
#define DXL_PROFILE_VELOCITY                60 
// #define DXL_PROFILE_VELOCITY_CONFIGCHANGE   70
#define DXL_PROFILE_ACCELERATION 5 // 적절한 가속도 값 설정
#define GOAL_VELOCITY_CALC_TA 0 // 속도 계산에 사용할 가속 시간 (ms)
#define DXL_VELOCITY_P_GAIN 100
#define DXL_VELOCITY_I_GAIN 1920

#define DXL1_ID                             1 
#define DXL2_ID                             2
#define DXL3_ID                             3 
#define DXL4_ID                             4 
#define DXL5_ID                             5
#define DXL_NUM                             5

#define BAUDRATE                            1000000 // 9600, 57600, 115200, 1000000, 2000000, 3000000, 4000000, 4500000
#define DRIVE_MODE_PROFILE                  1 // 0: Velocity Based Profile, 1: Time Based Profile
#define OPERATING_MODE                      4 // 1: Velocity Control Mode, 3: Position Control Mode, 4: Extended Position Control Mode, 16: PWM Control Mode
#define RETURN_DELAY_TIME                   0

#define DEVICENAME                          "/dev/ttyUSB0"  //sudo chmod a+rw /dev/ttyUSB0      //ls /dev/ttyUSB*   //pkill -f push_to_talk_app.py
#define AUDIO_DEVICE                        "hw:1"

//DXL initial goal position
// #define DEFAULT_PITCH                       3540
// #define DEFAULT_ROLL_R                      2560
// #define DEFAULT_ROLL_L                      1970
// #define DEFAULT_YAW                         1000
// #define DEFAULT_MOUTH                       1120

extern int DEFAULT_PITCH;
extern int DEFAULT_ROLL_R;
extern int DEFAULT_ROLL_L;
extern int DEFAULT_YAW;
extern int DEFAULT_MOUTH;

//robot parameter
#define PULLY_DIAMETER                      50
#define ROBOT_HEIGHT                        110             // 베이스부터 실이 연결된 레이어 까지의 높이 small -> 100,Large -> 180
#define ROBOT_HOLE_RADIUS                   25              // 로봇 머리 구멍 반지름 small -> 25, Large -> 50
#define ROBOT_YAW_GEAR_RATIO                2               // yaw 모터가 direct하게 머리를 회전시킨다면 1로 설정 아니면 2
#define ROBOT_MOUTH_TUNE                    60              // 최대 mouse movement size in DXL dimension -> 최초값에서 입모터 조정해보면서 결정
#define ROBOT_MOUTH_BACK_COMPENSATION       1.0             // 입 움직임에 대한 뒷쪽 보상 -> TRO 논문 참조  small -> 1.2, Large -> 1.5
#define ROBOT_MOUTH_PITCH_COMPENSATION      0.8             // 입 움직임에 따른 pitch 보상

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

int calculateDXLGoalVelocity_velocityBased(double current_position, double goal_position, double current_velocity, double profile_acceleration, double control_ms);

int calculateDXLGoalVelocity_timeBased_ds(double current_position, double goal_position, double current_velocity, double profile_acceleration, double control_ms);

bool readDXLPresentState(dynamixel::GroupBulkRead &groupBulkRead, int DXL_ID[], int present_velocity[], int present_position[]);

bool readDXLPresentVelocity(dynamixel::GroupSyncRead& groupSyncReadVelocity, int DXL_ID[], int present_velocity[]);

bool readDXLPresentPosition(dynamixel::GroupSyncRead& groupSyncReadPosition, int DXL_ID[], int present_position[]);

bool moveDXLwithVelocity(dynamixel::GroupSyncWrite& groupSyncWriteVelocity, int DXL_ID[], int goal_velocity[]);

bool moveDXLtoDesiredPosition(dynamixel::GroupSyncWrite& groupSyncWriteVelocity, dynamixel::GroupSyncWrite& groupSyncWritePosition, int DXL_ID[], int goal_position[], int velocity);

void update_DXL_goal_position(int DXL_goal_position[], int DXL_1, int DXL_2, int DXL_3, int DXL_4, int DXL_5);

//int를 32bit bin로 변경 (little endian)
void trans_int2bin_4(uint8_t param_goal_position[4], int dxl_goal_position);

int enable_torque(dynamixel::PacketHandler *packetHandler, dynamixel::PortHandler *portHandler, int *DXL_ID, uint8_t dxl_error);

int disable_torque(dynamixel::PacketHandler *packetHandler, dynamixel::PortHandler *portHandler, int *DXL_ID, uint8_t dxl_error);

int set_drive_mode_profile(dynamixel::PacketHandler *packetHandler, dynamixel::PortHandler *portHandler, int *DXL_ID, uint8_t dxl_error, int drive_mode_profile);

int set_operating_mode(dynamixel::PacketHandler *packetHandler, dynamixel::PortHandler *portHandler, int *DXL_ID, uint8_t dxl_error, int operating_mode);

int set_baudrate(dynamixel::PacketHandler *packetHandler, dynamixel::PortHandler *portHandler, int *DXL_ID, uint8_t dxl_error, int baudrate);

int set_return_delay_time(dynamixel::PacketHandler *packetHandler, dynamixel::PortHandler *portHandler, int *DXL_ID, uint8_t dxl_error, int return_delay_time);

int set_profile_acceleration(dynamixel::GroupSyncWrite &groupSyncWriteAcceleration, int *DXL_ID, int profile_acceleration);

int set_initial_gain (dynamixel::GroupSyncWrite &groupSyncWritePGain, dynamixel::GroupSyncWrite &groupSyncWriteIGain, int *DXL_ID, int profile_acceleration, int velocity_p_gain, int velocity_i_gain);

int assignClassWith1DMiddleBoundary(double x, const vector<double>& boundaries);

double calculateRMS(const vector<float>& data, size_t start, size_t frame_length) ;

double getSegmentAverageGrad(const vector<float>& data, const string& delta = "one2one", const string& mode = "abs");

vector<vector<double>> getNextSegment_SegSeg(const vector<double>& PrevEndOneBefore, const vector<double>& PrevEnd,const cnpy::NpyArray& segment,bool gradient = true, bool gotoZero = true);

vector<vector<double>> multExpToSegment(const vector<float>& ex_energy,vector<vector<double>> ex_segment,float threshold,float div);

Eigen::VectorXd toEigenVector(const vector<double>& stdVec);

vector<vector<double>> connectTwoSegments(const vector<vector<double>>& prevSegment, const vector<vector<double>>& nextSegment, int n_new, int n_anchor_past, int n_anchor_future);

float AM_fun(float min_open, float B, float r_k, float r_k_1, float r_k_2, float lim_delta_r);

std::tuple<float, float, float> lin_fit_fun2(float S, float X_pre, float grad_up_pre, float grad_down_pre, float del_grad, float dt);

