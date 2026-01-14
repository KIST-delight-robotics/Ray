#include "MotionLogger.h"

std::string create_log_directory(const std::string& base_dir) {
    auto now = std::chrono::system_clock::now();
    std::time_t t = std::chrono::system_clock::to_time_t(now);
    
    std::stringstream ss;
    // 예: output/motion_log/20231215_173000/
    ss << base_dir << std::put_time(std::localtime(&t), "%Y%m%d_%H%M%S");
    
    std::string path_str = ss.str();
    
    // 디렉토리 실제 생성
    try {
        std::filesystem::create_directories(path_str);
    } catch (const std::filesystem::filesystem_error& e) {
        std::cerr << "Error creating directory: " << e.what() << std::endl;
    }
    
    // 경로 끝에 슬래시 보장
    if (path_str.back() != '/') {
        path_str += "/";
    }
    
    return path_str;
}

// ==========================================
// DataLogger 구현 (40ms)
// ==========================================

DataLogger::DataLogger() {}

DataLogger::~DataLogger() {
    stop();
}

void DataLogger::start(const std::chrono::time_point<std::chrono::high_resolution_clock>& start_t, const std::string& dir) {
    std::lock_guard<std::mutex> lock(logMutex);
    if (is_open) return;

    auto now = std::chrono::system_clock::now();
    std::time_t t = std::chrono::system_clock::to_time_t(now);
    std::stringstream ss;
    ss << dir << "Standard_Log.csv";

    logFile.open(ss.str());
    // 헤더 작성
    logFile << "Timestamp_ms,Mode,"
            << "Target_Roll,Target_Pitch,Target_Yaw,Target_Mouth,"
            << "Target_Pos_0,Target_Pos_1,Target_Pos_2,Target_Pos_3,Target_Pos_4,"
            << "Present_Pos_0,Present_Pos_1,Present_Pos_2,Present_Pos_3,Present_Pos_4,"
            << "Present_Cur_0,Present_Cur_1,Present_Cur_2,Present_Cur_3,Present_Cur_4\n";
    
    session_start_time = start_t;
    is_open = true;
    std::cout << "[DataLogger] Started logging to " << ss.str() << std::endl;
}

void DataLogger::log(const std::string& mode,
                     const double* target_rpy,
                     const std::vector<int32_t>& target_pos,
                     const std::vector<MotorState>& states) {
    if (!is_open) return;

    std::lock_guard<std::mutex> lock(logMutex);
    auto now = std::chrono::high_resolution_clock::now();
    double elapsed_ms = std::chrono::duration<double, std::milli>(now - session_start_time).count();

    logFile << elapsed_ms << "," << mode;
    for(int i=0; i<4; ++i) logFile << "," << target_rpy[i];
    for(auto val : target_pos) logFile << "," << val;
    for(const auto& s : states) logFile << "," << s.position;
    for(const auto& s : states) logFile << "," << s.current;
    logFile << "\n";
}

void DataLogger::stop() {
    std::lock_guard<std::mutex> lock(logMutex);
    if (is_open) {
        logFile.close();
        is_open = false;
        std::cout << "[DataLogger] Logging stopped." << std::endl;
    }
}


// ==========================================
// HighFreqLogger 구현 (5ms)
// ==========================================

HighFreqLogger::HighFreqLogger(DynamixelDriver* drv) : driver(drv) {
    buffer.reserve(1000000); // 초기 버퍼 용량 예약
}

HighFreqLogger::~HighFreqLogger() {
    stop();
}

void HighFreqLogger::start(const std::chrono::time_point<std::chrono::high_resolution_clock>& start_t, const std::string& dir) {
    if (is_logging) return;
    
    shared_start_time = start_t;
    save_dir = dir;
    buffer.clear();
    is_logging = true;
    
    log_thread = std::thread(&HighFreqLogger::loop, this);
    std::cout << "[HighFreqLogger] Started." << std::endl;
}

void HighFreqLogger::stop() {
    if (!is_logging) return;
    is_logging = false;
    if (log_thread.joinable()) log_thread.join();
    saveToFile();
}

void HighFreqLogger::loop() {
    std::vector<MotorState> states;
    // 1ms 주기 설정
    auto interval = std::chrono::milliseconds(5);
    auto next_wake = std::chrono::high_resolution_clock::now();

    while (is_logging) {
        next_wake += interval; // 다음 깨어날 시간 갱신

        auto loop_start = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double, std::milli>(loop_start - shared_start_time).count();

        // 데이터 읽기
        if (driver->readAllState(states)) {
            HighFreqData data;
            data.timestamp_ms = elapsed;
            data.states = states;
            buffer.push_back(data);
        }

        // 남은 시간만큼 대기
        std::this_thread::sleep_until(next_wake);
    }
}

void HighFreqLogger::saveToFile() {
    if (buffer.empty()) return;

    auto now = std::chrono::system_clock::now();
    std::time_t t = std::chrono::system_clock::to_time_t(now);
    std::stringstream ss;
    
    ss << save_dir << "HighFreq_Log.csv";

    std::ofstream file(ss.str());
    if (!file.is_open()) {
        std::cerr << "[HighFreqLogger] Failed to open file: " << ss.str() << std::endl;
        return;
    }

    // 헤더
    file << "Timestamp_ms";
    for(int i=0; i<5; ++i) file << ",Present_Pos_" << i;
    for(int i=0; i<5; ++i) file << ",Present_Vel_" << i;
    for(int i=0; i<5; ++i) file << ",Present_Cur_" << i;
    file << "\n";

    // 데이터
    for (const auto& d : buffer) {
        file << d.timestamp_ms;
        for(const auto& s : d.states) file << "," << s.position;
        for(const auto& s : d.states) file << "," << s.velocity;
        for(const auto& s : d.states) file << "," << s.current;
        file << "\n";
    }
    
    file.close();
    std::cout << "[HighFreqLogger] Saved " << buffer.size() << " samples to " << ss.str() << std::endl;
}