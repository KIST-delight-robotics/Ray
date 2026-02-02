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
#include "Config.h"

using namespace std;
using namespace Eigen;

#define FORMAT              SF_FORMAT_WAV | SF_FORMAT_PCM_16
#define CHUNK_SIZE          1024

#define MAX_MOUTH           0 // 예시로 임의의 값 설정
#define MIN_MOUTH           300 // 예시로 임의의 값 설정

#define DXL_NUM             5

//MATH
#define COMPENSATION_MAX                    0.6
#define INCLINATION                         2               //볼륨조절함수에 쓸 기울기 
#define PI                                  3.14

//stream
#define INFO_STREAM( stream ) 
//std::cout << stream <<std::endl


struct RobotHomePose {
    int32_t home_pitch;
    int32_t home_roll_r;
    int32_t home_roll_l;
    int32_t home_yaw;
    int32_t home_mouth;
};

inline RobotHomePose g_home;


//파일 경로 생성
void create_file_path(std::string &file_path, const char *filename, const char *input_path);

void save_to_csv_float(const std::string& base_filename, const std::vector<float>& data, int frame_count);

void save_one_float_to_csv(const std::string& filename, float value);

void save_audio_file(const std::string& filename, const float* audiodata, sf_count_t frames, int samplerate, int channels);

std::pair<float, size_t> find_peak(const std::vector<float>& audio_buffer);

float calculate_mouth(float env_01, float max_MOUTH, float min_MOUTH) {
    // env_01: 0~1
    return min_MOUTH + env_01 * (max_MOUTH - min_MOUTH);
}


void save_audio_segment(const std::string& outputFilePath, const std::vector<float>& audioData, size_t dataSize);

std::vector<float> divide_channel(const std::vector<float>& audio_data, int channels, int frames);

float moving_average(const std::deque<float>& window);

float update_final_result(std::deque<float>& moving_average_window, size_t window_size, float new_sample);

float scale_max_sample(float max_sample);

float volume_control(std::deque<float>& recent_samples, size_t window_size, float new_sample);

float scaled_result_with_moving_average(std::deque<float>& recent_samples, size_t window_size, float new_sample);

std::vector<int32_t> RPY2DXL(double roll_f, double pitch_f, double yaw_f, double mouth_f, int mode);

int calculateDXLGoalVelocity_velocityBased(double current_position, double goal_position, double current_velocity, double profile_acceleration, double control_ms);

int calculateDXLGoalVelocity_timeBased_ds(double current_position, double goal_position, double current_velocity, double profile_acceleration, double control_ms);

void update_DXL_goal_position(int DXL_goal_position[], int DXL_1, int DXL_2, int DXL_3, int DXL_4, int DXL_5);

int assignClassWith1DMiddleBoundary(double x, const vector<double>& boundaries);

double calculateRMS(const vector<float>& data, size_t start, size_t frame_length) ;

double getSegmentAverageGrad(const vector<float>& data, const string& delta = "one2one", const string& mode = "abs");

vector<vector<double>> getNextSegment_SegSeg(const vector<double>& PrevEndOneBefore, const vector<double>& PrevEnd,const cnpy::NpyArray& segment,bool gradient = true, bool gotoZero = true);

vector<vector<double>> multExpToSegment(const vector<float>& ex_energy,vector<vector<double>> ex_segment,float threshold,float div);

Eigen::VectorXd toEigenVector(const vector<double>& stdVec);

vector<vector<double>> connectTwoSegments(const vector<vector<double>>& prevSegment, const vector<vector<double>>& nextSegment, int n_new, int n_anchor_past, int n_anchor_future);

float AM_fun(float min_open, float B, float r_k, float r_k_1, float r_k_2, float lim_delta_r);

std::tuple<float, float, float> lin_fit_fun2(float S, float X_pre, float grad_up_pre, float grad_down_pre, float del_grad, float dt);

// === Mouth Attack-Release Envelope 상태 ===
struct MouthEnvARState {
    double fs;
    double hpf_fc;
    double attack_ms;
    double release_ms;
    double T_open;
    double T_close;
    double gain;

    double a_hpf;
    double a_att;
    double a_rel;

    double prev_x;
    double prev_y;
    double env_prev;
    bool   talking;
};

// 초기화
void initMouthEnvAR(MouthEnvARState& st,
                    double fs,
                    double hpf_fc    = 90.0,
                    double attack_ms = 20.0,
                    double release_ms= 120.0,
                    double T_open    = 0.03,
                    double T_close   = 0.02,
                    double gain      = 10.0);

// 샘플 한 개 처리 → 0~1 mouth scaler
float processMouthEnvAR(MouthEnvARState& st, float x_in);
