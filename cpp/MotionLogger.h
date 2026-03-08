#ifndef MOTION_LOGGER_H
#define MOTION_LOGGER_H

#ifdef MOTOR_ENABLED
#include "DynamixelDriver.h" // MotorState 구조체 사용을 위해 필요
#else
// MOTOR_ENABLED가 아닐 때 MotorState 스텁 제공
#include <cstdint>
struct MotorState {
    int32_t position = 0;
    int32_t velocity = 0;
    int16_t current = 0;
};
#endif

#include <fstream>
#include <vector>
#include <string>
#include <mutex>
#include <atomic>
#include <thread>
#include <chrono>
#include <filesystem>
#include <sstream>
#include <iostream>
#include <iomanip>

std::string create_log_directory(const std::string& base_dir = "output/motion_log/");

// --- 1. 일반 로거 (40ms 주기) ---
class DataLogger {
private:
    std::ofstream logFile;
    std::mutex logMutex;
    std::chrono::time_point<std::chrono::high_resolution_clock> session_start_time;
    bool is_open = false;

public:
    DataLogger();
    ~DataLogger();

    void start(const std::chrono::time_point<std::chrono::high_resolution_clock>& start_t, const std::string& dir);
    void stop();

    // int32_t, MotorState 등을 사용하는 버전으로 수정
    void log(const std::string& mode,
             const double* target_rpy,
             const std::vector<int32_t>& target_pos,
             const std::vector<MotorState>& states);
};

// --- 2. 고속 로거 (5ms 주기, 하드웨어 전용) ---
#ifdef MOTOR_ENABLED
struct HighFreqData {
    double timestamp_ms;
    std::vector<MotorState> states;
};

class HighFreqLogger {
private:
    std::vector<HighFreqData> buffer;
    std::atomic<bool> is_logging{false};
    std::thread log_thread;
    std::chrono::time_point<std::chrono::high_resolution_clock> shared_start_time;

    DynamixelDriver* driver; // 드라이버 포인터 필요
    std::string save_dir;

    void loop(); // 내부 스레드 함수
    void saveToFile();

public:
    HighFreqLogger(DynamixelDriver* drv);
    ~HighFreqLogger();

    void start(const std::chrono::time_point<std::chrono::high_resolution_clock>& start_t, const std::string& dir);
    void stop();
};
#endif // MOTOR_ENABLED

#endif // MOTION_LOGGER_H