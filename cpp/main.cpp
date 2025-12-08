// 필요한 헤더 파일 포함
#include <iostream>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <chrono>
#include <SFML/Audio.hpp>
#include <iomanip>
#include <queue>
#include <vector>
#include <future>
#include <deque>
#include <cmath>
#include <sndfile.h> // 오디오 파일 입출력을 위한 헤더
#include <fstream>
#include <cstdlib>
#include <cstdint>
#include <ctime>
#include <algorithm>
#include <tuple>
#include <sstream>
#include <wiringPiI2C.h>
#include <wiringPi.h>
#include <unistd.h>
#include "cnpy.h"
#include "Macro_function.h"
#include <csignal>

// WebSocket 및 JSON 관련 헤더
#include <ixwebsocket/IXNetSystem.h>
#include <ixwebsocket/IXWebSocket.h>
#include <ixwebsocket/IXBase64.h> // Base64 디코딩을 위해 추가
#include <nlohmann/json.hpp>

using json = nlohmann::json;

#define MOTOR_ENABLED // 모터 연결 없이 테스트하려면 주석 처리

#define INTERVAL_MS 360 // 시퀀스 1개 당 시간
#define CONTROL_MS 40 // 모터 제어 주기
#define AUDIO_SAMPLE_RATE 24000
#define AUDIO_CHANNELS 1
#define MPU6050_ADDR 0x68
int fd;

int DEFAULT_PITCH = 3540 + 50;
int DEFAULT_ROLL_R = 2560 - 100;
int DEFAULT_ROLL_L = 2070 - 100;
int DEFAULT_YAW = 1000;
int DEFAULT_MOUTH = 1000 - 50;

sf::Music music;

dynamixel::PortHandler *portHandler;
dynamixel::PacketHandler *packetHandler;

// 파일 경로 설정
const std::string ASSETS_DIR = "assets";
const std::string DATA_DIR = "data";
const std::string MUSIC_DIR = ASSETS_DIR + "/audio/music";
const std::string VOCAL_DIR = ASSETS_DIR + "/audio/vocal";
const std::string SEGMENTS_DIR = DATA_DIR + "/segments";
const std::string IDLE_MOTION_FILE = ASSETS_DIR + "/headMotion/empty_10min.csv";
const std::string HEAD_GESTURE_FILE = ASSETS_DIR + "/headMotion/가까운 듯 먼 그대여.csv";
const std::string MOUTH_MOTION_FILE = ASSETS_DIR + "/mouthMotion/가까운 듯 먼 그대여-delta-big.csv";

// 전역 변수 및 동기화 도구
std::string vocal_file_path;

std::chrono::time_point<std::chrono::high_resolution_clock> start_time; // 쓰레드 대기 시간 설정용
double STT_DONE_TIME = 0.0; // STT 완료 시간 (사용자 입력 완료 후 음성 출력까지의 시간 측정용)

std::atomic<bool> stop_flag(false);
std::atomic<bool> user_interruption_flag(false);
std::atomic<bool> is_speaking(false);

int first_move_flag = 1;
float final_result = 0.0f;

std::queue<std::vector<float>> audio_queue;
std::mutex audio_queue_mutex;
std::condition_variable audio_queue_cv;

std::queue<std::pair<int, float>> mouth_motion_queue; // 사이클 번호와 모션 값 저장 (mouthmotion)
std::queue<std::vector<std::vector<double>>> head_motion_queue; // 슬라이스 저장 및 전달 (headmotion)
std::mutex mouth_motion_queue_mutex;
std::condition_variable mouth_motion_queue_cv;

int DXL_goal_position[DXL_NUM] = {0, 0, 0, 0, 0};
int DXL_past_position[DXL_NUM] = {0, 0, 0, 0, 0};
int DXL_goal_velocity[DXL_NUM] = {0, 0, 0, 0, 0};
// 상태를 읽어와 저장하는 배열
int DXL_present_position[DXL_NUM] = {0, 0, 0, 0, 0};
int DXL_present_velocity[DXL_NUM] = {0, 0, 0, 0, 0};

// 모션 보간을 위한 이전 값 저장
const int MAX_PREV_VALUES = 3;
std::deque<std::vector<double>> prevValues(MAX_PREV_VALUES, std::vector<double>(4, 0.0)); // 최근 3개의 값 저장
std::mutex prev_values_mutex; // prevValues 접근 동기화용 뮤텍스

// 로그 출력을 위한 뮤텍스
std::mutex cout_mutex;

std::atomic<bool> wait_mode_flag{false}; // true: on, false: off
bool music_flag = 0;
bool playing_music_flag = 0;

bool finish_adjust_ready = false;

// WebSocket 관련 전역 객체
ix::WebSocket webSocket;
std::queue<json> server_message_queue;
std::mutex server_message_queue_mutex;
std::condition_variable server_message_queue_cv;
std::promise<void> server_ready_promise;

// 스트리밍 데이터 처리를 위한 전역 변수
std::atomic<bool> is_realtime_streaming(false);
std::vector<uint8_t> realtime_stream_buffer;
std::mutex realtime_stream_buffer_mutex;
std::condition_variable realtime_stream_buffer_cv;

std::atomic<bool> is_responses_streaming(false);
std::vector<uint8_t> responses_stream_buffer;
std::mutex responses_stream_buffer_mutex;
std::condition_variable responses_stream_buffer_cv;

// 시간 포매터 함수
std::string get_time_str() {
    auto now = std::chrono::high_resolution_clock::now();
    auto now_c = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    std::stringstream ss;
    ss << std::put_time(std::localtime(&now_c), "%H:%M:%S");
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()) % 1000;
    ss << '.' << std::setfill('0') << std::setw(3) << ms.count();
    return ss.str();
}

// 쓰레드가 INTERVAL_MS 주기로 동작하게 하는 함수
void wait_for_next_cycle(int cycle_num) {
    auto next_cycle_time = start_time + std::chrono::milliseconds(INTERVAL_MS * cycle_num);
    std::this_thread::sleep_until(next_cycle_time);
}

// 모터 로깅을 위한 DataLogger 클래스
class DataLogger {
private:
    std::ofstream logFile;
    std::mutex logMutex;
    std::chrono::time_point<std::chrono::high_resolution_clock> session_start_time;
    bool is_open = false;
    std::string saveDirectory = "";

public:
    DataLogger() {}

    void createNewLogDir() {
        std::lock_guard<std::mutex> lock(logMutex);
        
        auto now = std::chrono::system_clock::now();
        std::time_t t = std::chrono::system_clock::to_time_t(now);
        std::stringstream ss;
        ss << "output/motion_log/" << std::put_time(std::localtime(&t), "%Y%m%d_%H%M%S");
        std::filesystem::path logDir = ss.str();
        std::filesystem::create_directories(logDir);
        saveDirectory = logDir.string() + "/";
        
        std::cout << "[Logger] Save directory set to: " << saveDirectory << std::endl;
    }

    void startSession() {
        std::lock_guard<std::mutex> lock(logMutex);
        if (is_open) return;

        // 현재 시간을 파일명으로 사용
        auto now = std::chrono::system_clock::now();
        std::time_t t = std::chrono::system_clock::to_time_t(now);
        std::stringstream ss;
        ss << saveDirectory << "MotionLog_" << std::put_time(std::localtime(&t), "%Y%m%d_%H%M%S") << ".csv";

        logFile.open(ss.str());
        // 헤더 작성 (Mode 컬럼 추가)
        logFile << "Timestamp_ms,Mode,"
                << "Target_Roll,Target_Pitch,Target_Yaw,Target_Mouth,"
                << "Target_Pos_0,Target_Pos_1,Target_Pos_2,Target_Pos_3,Target_Pos_4,"
                << "Present_Pos_0,Present_Pos_1,Present_Pos_2,Present_Pos_3,Present_Pos_4,"
                << "Present_Vel_0,Present_Vel_1,Present_Vel_2,Present_Vel_3,Present_Vel_4\n";
        
        session_start_time = std::chrono::high_resolution_clock::now();
        is_open = true;
        std::cout << "[Logger] Started logging to " << ss.str() << std::endl;
    }

    void log(const std::string& mode,
             const double* target_rpy,
             const int* target_pos, 
             const int* present_pos, 
             const int* present_vel) {
        if (!is_open) return;

        std::lock_guard<std::mutex> lock(logMutex);
        auto now = std::chrono::high_resolution_clock::now();
        double elapsed_ms = std::chrono::duration<double, std::milli>(now - session_start_time).count();

        logFile << elapsed_ms << "," << mode;
        
        // Target RPY (4 values)
        for(int i=0; i<4; ++i) logFile << "," << target_rpy[i];
        // Target Positions (5 motors)
        for(int i=0; i<5; ++i) logFile << "," << target_pos[i];
        // Present Positions (5 motors)
        for(int i=0; i<5; ++i) logFile << "," << present_pos[i];
        // Present Velocities (5 motors)
        for(int i=0; i<5; ++i) logFile << "," << present_vel[i];
        
        logFile << "\n";
    }

    void closeSession() {
        std::lock_guard<std::mutex> lock(logMutex);
        if (is_open) {
            logFile.close();
            is_open = false;
            std::cout << "[Logger] Session closed." << std::endl;
        }
    }
};

DataLogger robotLogger;

// **CustomSoundStream 클래스 정의**
class CustomSoundStream : public sf::SoundStream {
public:
    CustomSoundStream(unsigned int channelCount, unsigned int sampleRate)
        : m_channelCount(channelCount), m_sampleRate(sampleRate) {
        initialize(channelCount, sampleRate);
    }

    void appendData(const std::vector<sf::Int16>& data) {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_samples.insert(m_samples.end(), data.begin(), data.end());
        m_condition.notify_one();
    }
    void clearBuffer() {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_samples.clear();  // 저장된 샘플 데이터 초기화
    }

protected:
    virtual bool onGetData(Chunk& data) override {
    std::unique_lock<std::mutex> lock(m_mutex);

        if (m_samples.empty()) {
            // stop_flag가 설정되었고 버퍼가 비었으면 스트림을 중지합니다.
            if (stop_flag) {
                return false;
            }

            // 버퍼에 데이터가 없을 때 무음 재생
            static std::vector<sf::Int16> silence(m_sampleRate * m_channelCount / 10, 0); // 0.1초 분량의 무음
            data.samples = silence.data();
            data.sampleCount = silence.size();
            return true;
        }

        // 재생할 최대 샘플 수 결정
        std::size_t sampleCount = std::min(m_samples.size(), static_cast<std::size_t>(m_sampleRate * m_channelCount *80 / 1000)); // 80msec 분량씩 데이터 가져감 

        // 재생할 샘플 설정
        m_chunkSamples.assign(m_samples.begin(), m_samples.begin() + sampleCount);
        data.samples = m_chunkSamples.data();       //SFML에게 직접 data 제공
        data.sampleCount = m_chunkSamples.size();

        // 재생한 샘플은 버퍼에서 제거
        m_samples.erase(m_samples.begin(), m_samples.begin() + sampleCount);

        
        return true;
    }

    virtual void onSeek(sf::Time timeOffset) override {
        // 시킹 기능이 필요한 경우 구현(스트림의 재생 위치를 변경해야 할 때 호출)
    }
private:
    std::vector<sf::Int16> m_samples;
    std::vector<sf::Int16> m_chunkSamples;
    unsigned int m_channelCount;
    unsigned int m_sampleRate;
    std::mutex m_mutex;
    std::condition_variable m_condition;
};

bool initialize_dynamixel() {
    portHandler = dynamixel::PortHandler::getPortHandler(DEVICENAME);
    packetHandler = dynamixel::PacketHandler::getPacketHandler(PROTOCOL_VERSION);

    if (!(portHandler->openPort())) {
        std::lock_guard<std::mutex> lock(cout_mutex);
        std::cerr << "Failed to open the port!" << std::endl;
        return false;
    }

    std::vector<int> rate_options = {9600, 57600, 115200, 1000000, 2000000, 3000000, 4000000, 4500000};
    int dxl_comm_result = COMM_TX_FAIL;             // 통신 결과
    uint8_t dxl_error = 0;                          // 모터 에러 상태

    for (int rate : rate_options) {
        if (!portHandler->setBaudRate(rate)) {
            printf("Failed to change the baudrate %d (Skipping)\n", rate);
            continue;
        }
        printf("Trying %d bps...\n", rate);

        dxl_comm_result = packetHandler->ping(portHandler, DXL1_ID, &dxl_error);

        if (dxl_comm_result == COMM_SUCCESS && dxl_error == 0) {
            printf("Success!\n");
            printf("Baudrate set to %d bps.\n", rate);
            break; // 성공적으로 통신이 이루어지면 루프 종료
        } else {
            printf("Failed.\n");
        }
    }

    int DXL_ID[DXL_NUM] = { DXL1_ID, DXL2_ID, DXL3_ID, DXL4_ID, DXL5_ID };

    if (!disable_torque(packetHandler, portHandler, DXL_ID, dxl_error)) {
        printf("Failed to torque disable\n");
        return false;
    } 
    else {
        // EEPROM 설정
        // Position 값을 절대위치 값으로 초기화하기 위해 Position Control Mode로 한번 설정 후 변경
        set_operating_mode(packetHandler, portHandler, DXL_ID, dxl_error, 3);
        set_operating_mode(packetHandler, portHandler, DXL_ID, dxl_error, OPERATING_MODE);
        set_drive_mode_profile(packetHandler, portHandler, DXL_ID, dxl_error, DRIVE_MODE_PROFILE);
        set_baudrate(packetHandler, portHandler, DXL_ID, dxl_error, BAUDRATE);
        set_return_delay_time(packetHandler, portHandler, DXL_ID, dxl_error, RETURN_DELAY_TIME);
        // RAM 설정
        dynamixel::GroupSyncWrite groupSyncWriteAcceleration(portHandler, packetHandler, ADDR_PRO_PROFILE_ACCELERATION, LEN_PRO_PROFILE_ACCELERATION);
        dynamixel::GroupSyncWrite groupSyncWritePositionPGain(portHandler, packetHandler, ADDR_PRO_POSITION_P_GAIN, LEN_PRO_POSITION_P_GAIN);
        dynamixel::GroupSyncWrite groupSyncWritePositionIGain(portHandler, packetHandler, ADDR_PRO_POSITION_I_GAIN, LEN_PRO_POSITION_I_GAIN);
        set_profile_acceleration(groupSyncWriteAcceleration, DXL_ID, DXL_PROFILE_ACCELERATION);

        int position_p_gain[DXL_NUM] = {460, 1000, 1000, 460, 460};
        int position_i_gain[DXL_NUM] = {0, 50, 50, 0, 0};
        set_position_gain(groupSyncWritePositionPGain, groupSyncWritePositionIGain, DXL_ID, position_p_gain, position_i_gain);
    }

    if (!enable_torque(packetHandler, portHandler, DXL_ID, dxl_error)) {
        printf("Failed to torque enable\n");
        return false;
    }

    printf("Motors initialized (Port Open, Torque On).\n");
    return true;
}

void updatePrevValues(double roll, double pitch, double yaw, double mouth) {
    // 이 함수에 들어오면 자물쇠를 잠금 (다른 쓰레드 대기)
    std::lock_guard<std::mutex> lock(prev_values_mutex);

    // 데이터 추가
    prevValues.push_back({roll, pitch, yaw, mouth});

    // n개 초과 시 앞부분 삭제
    while (prevValues.size() > MAX_PREV_VALUES) {
        prevValues.pop_front();
    }

    // 함수가 끝나면 lock 변수가 사라지면서 자동으로 자물쇠가 풀림(Unlock)
}

std::vector<std::vector<double>> applyOffsetDecay(
    const std::vector<double>& startPose,
    std::vector<std::vector<double>> targetTraj,
    int blend_frames)
{
    // 궤적이 비었거나 사이즈가 안 맞으면 그대로 반환
    if (targetTraj.empty() || startPose.size() != targetTraj[0].size()) {
		std::cout << "applyOffsetDecay: Invalid input sizes." << std::endl;
        return targetTraj;
    }

    // 보정 프레임 수가 궤적 길이보다 길면 궤적 길이만큼만 적용
    if (blend_frames > targetTraj.size()) {
		std::cout << "applyOffsetDecay: blend_frames exceeds trajectory size. Adjusting blend_frames to target trajectory's size." << std::endl;
        blend_frames = targetTraj.size();
    }

    // 초기 오프셋(차이) 계산: (현재 위치) - (궤적의 첫 위치)
    std::vector<double> diffs;
    for (size_t j = 0; j < startPose.size(); ++j) {
        diffs.push_back(startPose[j] - targetTraj[0][j]);
    }

    // 오프셋 감쇄 적용
    for (int i = 0; i < blend_frames; ++i) {
        // t: 0.0 ~ 1.0 (마지막 프레임에서 1.0 도달)
        double t = (double)(i + 1) / blend_frames;

        // Smoothstep (S자 곡선)
        double alpha = t * t * (3.0 - 2.0 * t);

        // Decay (1.0 -> 0.0)
        double decay = 1.0 - alpha;

        // 각 값(Roll, Pitch, Yaw, Mouth)에 대해 보정값 적용
        for (size_t j = 0; j < targetTraj[i].size(); ++j) {
            targetTraj[i][j] += diffs[j] * decay;
        }
    }

    return targetTraj;
}

void move_to_initial_position_posctrl() {
    // GroupSyncWrite 인스턴스 초기화
    dynamixel::GroupSyncWrite groupSyncWritePosition(portHandler, packetHandler, ADDR_PRO_GOAL_POSITION, LEN_PRO_GOAL_POSITION);
    dynamixel::GroupSyncWrite groupSyncWriteProfileVelocity(portHandler, packetHandler, ADDR_PRO_PROFILE_VELOCITY, LEN_PRO_PROFILE_VELOCITY);

    int DXL_ID[DXL_NUM] = { DXL1_ID, DXL2_ID, DXL3_ID, DXL4_ID, DXL5_ID };
    int DXL_initial_position[DXL_NUM] = { DEFAULT_PITCH, DEFAULT_ROLL_R, DEFAULT_ROLL_L, DEFAULT_YAW, DEFAULT_MOUTH };

    // 초기 위치로 이동
    moveDXLtoDesiredPosition(groupSyncWriteProfileVelocity, groupSyncWritePosition, DXL_ID, DXL_initial_position, DXL_PROFILE_VELOCITY_HOMING);

    // 모터 위치 업데이트
    for (int i = 0; i < DXL_NUM; i++) {
        DXL_past_position[i] = DXL_initial_position[i];
    }
}

void move_to_initial_position_velctrl() {
    dynamixel::GroupSyncRead groupSyncReadPosition(portHandler, packetHandler, ADDR_PRO_PRESENT_POSITION, LEN_PRO_PRESENT_POSITION);
    dynamixel::GroupSyncWrite groupSyncWriteGoalVelocity(portHandler, packetHandler, ADDR_PRO_GOAL_VELOCITY, LEN_PRO_GOAL_VELOCITY);

    int DXL_ID[DXL_NUM] = { DXL1_ID, DXL2_ID, DXL3_ID, DXL4_ID, DXL5_ID };
    int DXL_initial_position[DXL_NUM] = { DEFAULT_PITCH, DEFAULT_ROLL_R, DEFAULT_ROLL_L, DEFAULT_YAW, DEFAULT_MOUTH };

    const int POSITION_TOLERANCE = 20; // 목표 위치 도달로 간주할 허용 오차
    const double P_GAIN = 0.2; // 비례 제어 상수 (이 값을 조절하여 감속 강도 변경)
    const int MAX_VELOCITY = 100; // 최대 속도 제한
    const int MIN_VELOCITY = 30;  // 최소 구동 속도

    printf("Moving to initial position...\n");

    while (true) {
        // 1. 현재 위치 읽기
        readDXLPresentPosition(groupSyncReadPosition, DXL_ID, DXL_present_position);

        bool all_motors_in_position = true;
        for (int i = 0; i < DXL_NUM; i++) {
            int position_diff = DXL_initial_position[i] - DXL_present_position[i];
            std::cout << "Motor " << DXL_ID[i] << ": Current Position = " << DXL_present_position[i]
                      << ", Target Position = " << DXL_initial_position[i]
                      << ", Diff = " << position_diff << std::endl;

            // 2. 목표 위치에 도달했는지 확인
            if (std::abs(position_diff) > POSITION_TOLERANCE) {
                all_motors_in_position = false;
                // 3. 목표 위치 방향으로 속도 설정
                int calculated_velocity = static_cast<int>(position_diff * P_GAIN);

                // 최대 속도 제한
                if (calculated_velocity > MAX_VELOCITY) {
                    calculated_velocity = MAX_VELOCITY;
                } else if (calculated_velocity < -MAX_VELOCITY) {
                    calculated_velocity = -MAX_VELOCITY;
                }
                // 최소 속도 보정 (목표 지점 근처에서 멈추는 현상 방지)
                if (calculated_velocity > 0 && calculated_velocity < MIN_VELOCITY) {
                    calculated_velocity = MIN_VELOCITY;
                } else if (calculated_velocity < 0 && calculated_velocity > -MIN_VELOCITY) {
                    calculated_velocity = -MIN_VELOCITY;
                }

                DXL_goal_velocity[i] = calculated_velocity;
            } else {
                DXL_goal_velocity[i] = 0; // 목표 도달 시 정지
            }
        }

        // 4. 계산된 목표 속도를 모터에 명령
        moveDXLwithVelocity(groupSyncWriteGoalVelocity, DXL_ID, DXL_goal_velocity);

        // 모든 모터가 목표 위치에 도달하면 루프 종료
        if (all_motors_in_position) {
            printf("Initial position reached.\n");
            break;
        }

        // 제어 주기 맞추기
        std::this_thread::sleep_for(std::chrono::milliseconds(CONTROL_MS));
    }

    // 최종적으로 모터 정지 명령
    int final_velocity[DXL_NUM] = {0, 0, 0, 0, 0};
    moveDXLwithVelocity(groupSyncWriteGoalVelocity, DXL_ID, final_velocity);
}

// 첫 번째 쓰레드: 오디오 스트림을 받아 분할합니다.
void stream_and_split(const SF_INFO& sfinfo, CustomSoundStream& soundStream, const std::string& stream_type) {
    // --- 스트림 타입에 따라 사용할 버퍼와 동기화 객체 선택 ---
    std::vector<uint8_t>* buffer;
    std::mutex* buffer_mutex;
    std::condition_variable* buffer_cv;
    std::atomic<bool>* is_streaming_flag;

    if (stream_type == "realtime") {
        buffer = &realtime_stream_buffer;
        buffer_mutex = &realtime_stream_buffer_mutex;
        buffer_cv = &realtime_stream_buffer_cv;
        is_streaming_flag = &is_realtime_streaming;
    } else { // "responses"
        buffer = &responses_stream_buffer;
        buffer_mutex = &responses_stream_buffer_mutex;
        buffer_cv = &responses_stream_buffer_cv;
        is_streaming_flag = &is_responses_streaming;
    }

    // --- 초기 설정 ---
    int channels = sfinfo.channels;
    int samplerate = sfinfo.samplerate;
    const size_t bytes_per_interval = samplerate * channels * sizeof(sf::Int16) * INTERVAL_MS / 1000;

    for (int cycle_num = -2; ; ++cycle_num) {
        if (user_interruption_flag) {
            std::cout << "Interruption detected in stream_and_split." << std::endl;
            break;
        }
        wait_for_next_cycle(cycle_num);

        // --- 1. 데이터 획득 ---
        std::vector<uint8_t> raw_chunk;
        {
            std::unique_lock<std::mutex> lock(*buffer_mutex);
            buffer_cv->wait(lock, [&] {
                return buffer->size() >= bytes_per_interval || !(*is_streaming_flag);
            });

            if (!(*is_streaming_flag) && buffer->empty()) {
                break;
            }

            size_t size_to_take = std::min(buffer->size(), bytes_per_interval);
            size_to_take -= size_to_take % (sizeof(sf::Int16) * channels);
            if (size_to_take == 0) continue;

            raw_chunk.assign(buffer->begin(), buffer->begin() + size_to_take);
            buffer->erase(buffer->begin(), buffer->begin() + size_to_take);
        }

        // --- 2. 데이터 가공 ---
        size_t num_samples = raw_chunk.size() / sizeof(sf::Int16);
        std::vector<sf::Int16> audio_for_playback(num_samples);
        std::vector<float> audio_for_motion(num_samples);

        for (size_t i = 0; i < num_samples; ++i) {
            sf::Int16 sample = static_cast<sf::Int16>(raw_chunk[i*2] | (raw_chunk[i*2 + 1] << 8));
            audio_for_playback[i] = sample;
            audio_for_motion[i] = static_cast<float>(sample) / 32767.0f;
        }

        // --- 3. 데이터 전달 ---
        soundStream.appendData(audio_for_playback);
        {
            std::lock_guard<std::mutex> lock(audio_queue_mutex);
            audio_queue.push(audio_for_motion);
        }
        audio_queue_cv.notify_one();

        {
            auto now = std::chrono::high_resolution_clock::now();
            std::lock_guard<std::mutex> lock(cout_mutex);
            std::cout << "Stream and split cycle " << cycle_num << " at "
                      << std::chrono::duration_cast<std::chrono::milliseconds>(now - start_time).count()
                      << " ms" << std::endl;
        }
    }

    // --- 4. 종료 처리 ---
    stop_flag = true;
    audio_queue_cv.notify_one();
}

// 첫 번째 쓰레드: 오디오 파일을 읽어 분할합니다.
void read_and_split(SNDFILE* sndfile, const SF_INFO& sfinfo, CustomSoundStream& soundStream) {
    // --- 초기 설정 ---
    int channels = sfinfo.channels;
    int samplerate = sfinfo.samplerate;
    int frames_per_interval = samplerate * INTERVAL_MS / 1000;
    sf_count_t total_frames = sfinfo.frames;
    sf_count_t position = 0;
    bool playback_started = false;

    std::vector<float> audio_buffer(frames_per_interval * channels);
    std::vector<float> vocal_buffer; // 필요할 때만 크기 할당

    // 음악 재생 시, 모션 생성은 보컬 파일 기준
    SNDFILE* vocal_sndfile = nullptr;
    if (playing_music_flag) {
        SF_INFO vocal_sfinfo;
        vocal_sndfile = sf_open(vocal_file_path.c_str(), SFM_READ, &vocal_sfinfo);
        if (!vocal_sndfile) {
            std::lock_guard<std::mutex> lock(cout_mutex);
            std::cerr << "Error: Vocal file not found at " << vocal_file_path << ". Aborting playback." << std::endl;
            stop_flag = true;
            audio_queue_cv.notify_one(); // 대기 중인 스레드를 깨워 즉시 종료
            return; // 함수 즉시 종료
        }
        vocal_buffer.resize(frames_per_interval * channels);
    }

    for (int cycle_num = -2; ; ++cycle_num) {
        if (user_interruption_flag) {
            std::cout << "Interruption detected in read_and_split." << std::endl;
            break;
        }
        wait_for_next_cycle(cycle_num);

        // --- 1. 데이터 획득 ---
        // 파일에서 주기(INTERVAL_MS)에 해당하는 오디오 데이터를 읽어옵니다.
        sf_seek(sndfile, position, SEEK_SET);
        sf_count_t frames_to_read = std::min((sf_count_t)frames_per_interval, total_frames - position);
        sf_count_t frames_read = sf_readf_float(sndfile, audio_buffer.data(), frames_to_read);

        if (frames_read == 0) {
            break; // 파일의 끝에 도달하면 루프 종료
        }
        audio_buffer.resize(frames_read * channels);

        // --- 2. 데이터 가공 ---
        // 획득한 메인 오디오 데이터를 재생용(Int16)으로 변환합니다.
        std::vector<sf::Int16> int16_data(audio_buffer.size());
        for (std::size_t i = 0; i < audio_buffer.size(); ++i) {
            int16_data[i] = static_cast<sf::Int16>(audio_buffer[i] * 32767);
        }

        // --- 3. 데이터 전달 ---
        // 재생용 데이터와 모션 생성용 데이터를 각각의 소비자에게 전달합니다.
        soundStream.appendData(int16_data);
        {
            std::lock_guard<std::mutex> lock(audio_queue_mutex);
            if (playing_music_flag && vocal_sndfile) {
                sf_count_t vocal_frames_read = sf_readf_float(vocal_sndfile, vocal_buffer.data(), frames_to_read);
                vocal_buffer.resize(vocal_frames_read * channels);
                audio_queue.push(vocal_buffer);
            } else {
                audio_queue.push(audio_buffer);
            }
        }
        audio_queue_cv.notify_one();

        position += frames_per_interval;

        {
            auto now = std::chrono::high_resolution_clock::now();
            std::lock_guard<std::mutex> lock(cout_mutex);
            std::cout << "Read and split cycle " << cycle_num << " at "
                      << std::chrono::duration_cast<std::chrono::milliseconds>(now - start_time).count()
                      << " ms" << std::endl;
        }
    }

    // --- 4. 종료 처리 ---
    // 모든 처리가 끝났음을 후속 스레드에 알립니다.
    stop_flag = true;
    audio_queue_cv.notify_one(); // 대기 중인 generate_motion 스레드를 깨워 종료 조건을 확인시킵니다.
    if (vocal_sndfile) sf_close(vocal_sndfile);
}


// 두 번째 쓰레드: 녹음본을 가지고 모션 생성
void generate_motion(int channels, int samplerate) {

    std::vector<float> audio_buffer;
    const size_t MOVING_AVERAGE_WINDOW_SIZE = 5;
    const size_t MAX_SAMPLE_WINDOW_SIZE = 40;
    std::deque<float> moving_average_window;

    while(!audio_queue.empty()) audio_queue.pop();      //인터럽트시 이미 차있는 오디오 큐 비어줘야함.

    int frames_per_update = samplerate * 40 / 1000; // 40ms에 해당하는 프레임 수
    int find_peak_point = samplerate * 36 / 1000; // 36ms에서 시작하기 위한 위치
    float v1_th = 0.05;
    float s_max = 0.4;
    int num_motion_size = 0;
    float min_open = 0.25;
    int B = 10;              // 끌어올리기 함수 기울기
    float lim_delta_r = 0.1; //AM funtion에서 사용
    float ex_v1_max_sc_avg = 0;
    float exx_v1_max_sc_avg = 0;
    float dt = 0.040;
    float del_grad = 0.1;
    float grad_up_pre = 0.0f;
    float grad_down_pre = 0.0f;
    float grad_up_now = 0.0f;
    float grad_down_now = 0.0f;
    float X_pre = 0.0f;

    std::vector<float> v1_max;

    std::vector<double> prevEndOneBefore = {0.0, 0.0, 0.0};
    std::vector<double> prevEnd = {0.0, 0.0, 0.0};
    std::vector<std::vector<double>> deliverSegment;
    std::vector<std::vector<double>> prevSegment;
    std::vector<double> boundaries = {0.01623224, 0.02907711, 0.04192197};

    int first_segment_flag = 1;

    for (int cycle_num = -1; ; ++cycle_num) {
        if (user_interruption_flag) {
            std::cout << "Interruption detected in generate_motion." << std::endl;
            break;
        }
        wait_for_next_cycle(cycle_num);
        if (stop_flag && audio_queue.empty()) {
            std::cout << "generate motion break ------------------------" << std::endl;
            break;
        }

        double avg_grad;
        int segClass;
        std::vector<float> energy;

        int num_motion_updates = INTERVAL_MS / 40;

        // 오디오 데이터 가져오기
        std::unique_lock<std::mutex> lock(audio_queue_mutex);
        audio_queue_cv.wait(lock, [] {return !audio_queue.empty() || stop_flag;});

        if (stop_flag && audio_queue.empty()) {
            std::cout << "generate motion break ------------------------" << std::endl;
            break;
        }

        audio_buffer = std::move(audio_queue.front());
        audio_queue.pop();
        lock.unlock();

        std::vector<float> motion_results;

        for(int i = 0; i < num_motion_updates; ++i) {

            num_motion_size++;

            // 시작 인덱스와 끝 인덱스 계산
            int start_frame = i * frames_per_update;
            int end_frame = start_frame + frames_per_update;

            int start_frame_mouth = i * frames_per_update + find_peak_point;
            int end_frame_mouth = i * frames_per_update + frames_per_update;

            // 범위 체크
            if (end_frame > audio_buffer.size() / channels) {
                end_frame = audio_buffer.size() / channels;
                //i = num_motion_updates -1;
            }
            if (start_frame_mouth >= audio_buffer.size() / channels) {
                // 마지막 구간이므로 더 이상 처리할 오디오가 없음
                // break 또는 return 등을 통해 해당 루프/함수를 빠져나감
                std::cout << "stop flag : " << stop_flag << ", audio queue size : " << audio_queue.size() << std::endl;
                break; 
            }
            // 범위 체크
            if (end_frame_mouth > audio_buffer.size() / channels) {
                end_frame_mouth = audio_buffer.size() / channels;
                //i = num_motion_updates -1;
            }

            // 현재 업데이트에 해당하는 오디오 데이터 추출
            std::vector<float> current_audio(audio_buffer.begin() + start_frame * channels,
                                             audio_buffer.begin() + end_frame * channels);

            std::vector<float> current_audio_mouth(audio_buffer.begin() + start_frame_mouth * channels,
                                             audio_buffer.begin() + end_frame_mouth * channels);
            // 채널 분리
            std::vector<float> channel_divided = divide_channel(current_audio, channels, end_frame - start_frame);

            // 채널 분리
            std::vector<float> channel_divided_mouth = divide_channel(current_audio_mouth, channels, end_frame_mouth - start_frame_mouth);

            // 최대값과 해당 인덱스 찾기
            auto [max_sample, max_index] = find_peak(channel_divided_mouth);

            if (max_sample < 0) max_sample = 0;
            
            v1_max.push_back(max_sample);
            

            //cp 사용시 아래 주석 해제 할 것.

            std::tie(max_sample, grad_up_now, grad_down_now) = lin_fit_fun2(max_sample, X_pre, grad_up_pre, grad_down_pre, del_grad, dt);
            X_pre = max_sample;
            grad_up_pre = grad_up_now;
            grad_down_pre = grad_down_now;

            ////////////////////////////////////////////////////

            // s_max 값 구하기 (s_max가 크면 scaling이 작아져 값이 전체적으로 작아지는 역할을 함)
            if(num_motion_size > 2 && max_sample > v1_th && v1_max[v1_max.size()-2] < v1_th){
                if(v1_max.size() < 100){
                    for(float i = 0; i< v1_max.size(); i++){
                        s_max = std::max(s_max, v1_max[i]);
                    }
                }
                else{
                    for(float i = v1_max.size() - 100; i< v1_max.size(); i++){
                        s_max = std::max(s_max,v1_max[i]);
                    }
                }
            }

            double sc = 0.4/s_max;

            //cout << "s_max : " << s_max << " sc: " << sc << '\n';
            //cout << "raw_sample : " << max_sample << '\n';
            max_sample = sc  * max_sample;
            //cout << "sc_sample : " << max_sample << '\n';


            // max_sample을 이전 5개값과 평균을 내서 평균값을 max_sample로 사용
            if(num_motion_size > 5) max_sample = update_final_result(moving_average_window, MOVING_AVERAGE_WINDOW_SIZE, max_sample); 
            else{
                update_final_result(moving_average_window, MOVING_AVERAGE_WINDOW_SIZE, max_sample);
            }

            //cout << "AVG_sample : " << max_sample << '\n';

            // max_sample 값이 min_open 값 이하일 때 하이퍼 탄젠트 적용해서 mouth 모션을 좀 더 역동적으로 만들어줌.
            if(num_motion_size > 5){
                if(max_sample > min_open) final_result = max_sample;
                else{
                    final_result = AM_fun(min_open,B, max_sample, ex_v1_max_sc_avg, exx_v1_max_sc_avg, lim_delta_r);
                }
            }else{
                max_sample = 0;
            }
            //cout << "ex_v1_max_sc_avg : " << ex_v1_max_sc_avg << ", exx_v1_max_sc_avg : " << exx_v1_max_sc_avg << '\n';
            exx_v1_max_sc_avg = ex_v1_max_sc_avg;
            ex_v1_max_sc_avg = max_sample;

            //cout << "final_result : " << final_result << '\n';
            float calculate_result = calculate_mouth(final_result, MAX_MOUTH, MIN_MOUTH);
            //cout<< "calculate result : " << calculate_result << '\n';   

            motion_results.push_back(calculate_result);
            
            // -- 헤드 모션 생성을 위한 energy 저장 --
            double rms_value = calculateRMS(channel_divided, 0, frames_per_update);
            energy.push_back(rms_value);

            // if(i == num_motion_updates - 1) { // 마지막 업데이트일 때만 헤드 모션 생성

            //     if(first_segment_flag == 1) {

            //         double start_mouth = 0.0;

            //         // 첫 세그먼트일 경우 prevSegment를 이전 값들로 초기화
            //         {
            //             std::lock_guard<std::mutex> lock(prev_values_mutex);
            //             prevSegment.clear();
            //             for (const auto& val : prevValues) {
            //                 prevSegment.push_back({val[0], val[1], val[2]});
            //             }
            //             start_mouth = prevValues.back()[3];
            //         }

            //         // // 말을 시작하는 시점의 입모양 보정
            //         // int blend_frames = 5; // 보정할 프레임 수 (5프레임 = 약 200ms)

            //         // for (int k = 0; k < blend_frames; ++k) {
            //         //     // 0.0 ~ 1.0 으로 증가하는 선형 비율 t
            //         //     double t = (double)(k + 1) / (double)(blend_frames);

            //         //     // Smoothstep 적용 (3차 곡선 효과)
            //         //     double alpha = t * t * (3.0 - 2.0 * t);

            //         //     // 보간: (시작값 * (1 - alpha)) + (목표값 * alpha)
            //         //     motion_results[k] = static_cast<float>(start_mouth * (1.0 - alpha) + motion_results[k] * alpha);
            //         // }
            //     }
                
            //     //평균 기울기 값 계산
            //     avg_grad = getSegmentAverageGrad(energy, "one2one" , "abs");

            //     // 평균 기울기 값이 4개 class 중 어디에 해당하는지 판단 
            //     segClass = assignClassWith1DMiddleBoundary(avg_grad, boundaries);
            //     //cout << "Assigned class : " << segClass << endl;
            //     std::string filePath;

            //     switch (segClass) {
            //         case 0: filePath =  "segment_0.npy"; break;
            //         case 1: filePath =  "segment_1.npy"; break;
            //         case 2: filePath =  "segment_2.npy"; break;
            //         case 3: filePath =  "segment_3.npy"; break;
            //         default:
            //             std::cerr << "Invalid segClass: " << segClass << std::endl;
            //             break;
            //     }

            //     cnpy::NpyArray segment = cnpy::npy_load(SEGMENTS_DIR + "/" + filePath);

            //     for (int j = 0; j < 3; j++){
            //         prevEnd[j] = prevSegment[prevSegment.size() -1][j]; // prevSegment의 마지막 데이터 값
            //         prevEndOneBefore[j] = prevSegment[prevSegment.size() -2][j];
            //     }

            //     //segment 선택
            //     deliverSegment = getNextSegment_SegSeg(prevEndOneBefore, prevEnd, segment, true, true);
            
            //     // segment 보정 (무성구간에 따라서 값 보정)
            //     deliverSegment = multExpToSegment(energy, deliverSegment, 0.01, 10);

			// 	if (first_segment_flag == 1) {
			// 		// 첫 세그먼트일 경우 이전 값들과 자연스럽게 이어지도록 보정
			// 		// std::vector<double> startPose = { prevEnd[0], prevEnd[1], prevEnd[2] };
			// 		// deliverSegment = applyOffsetDecay(startPose, deliverSegment, 9); // 5프레임 동안 보정 적용
			// 		first_segment_flag = 0;
            //     }
            //     else {
            //     // 이전 Segment의 끝부분과 현재 Segment의 시작부분을 B-spline 보간법을 통해 자연스럽게 이어줌.
            //         deliverSegment = connectTwoSegments(prevSegment, deliverSegment, 3, 5, 3);
            //     }

            //     // 현재 세그먼트를 다음 반복을 위해 저장
            //     prevSegment = deliverSegment;
            // }
        }

        if(!energy.empty()) { // 마우스 모션 생성 완료 후 마지막에 한번만 헤드 모션 생성

            if(first_segment_flag == 1) {

                double start_mouth = 0.0;

                // 첫 세그먼트일 경우 prevSegment를 이전 값들로 초기화
                {
                    std::lock_guard<std::mutex> lock(prev_values_mutex);
                    prevSegment.clear();
                    for (const auto& val : prevValues) {
                        prevSegment.push_back({val[0], val[1], val[2]});
                    }
                    start_mouth = prevValues.back()[3];
                }

                // 말을 시작하는 시점의 입모양 보정
                int blend_frames = 5; // 보정할 프레임 수 (5프레임 = 약 200ms)

                for (int k = 0; k < blend_frames; ++k) {
                    // 0.0 ~ 1.0 으로 증가하는 선형 비율 t
                    double t = (double)(k + 1) / (double)(blend_frames);

                    // Smoothstep 적용 (3차 곡선 효과)
                    double alpha = t * t * (3.0 - 2.0 * t);

                    // 보간: (시작값 * (1 - alpha)) + (목표값 * alpha)
                    motion_results[k] = static_cast<float>(start_mouth * (1.0 - alpha) + motion_results[k] * alpha);
                }
            }
            
            //평균 기울기 값 계산
            avg_grad = getSegmentAverageGrad(energy, "one2one" , "abs");

            // 평균 기울기 값이 4개 class 중 어디에 해당하는지 판단 
            segClass = assignClassWith1DMiddleBoundary(avg_grad, boundaries);
            //cout << "Assigned class : " << segClass << endl;
            std::string filePath;

            switch (segClass) {
                case 0: filePath =  "segment_0.npy"; break;
                case 1: filePath =  "segment_1.npy"; break;
                case 2: filePath =  "segment_2.npy"; break;
                case 3: filePath =  "segment_3.npy"; break;
                default:
                    std::cerr << "Invalid segClass: " << segClass << std::endl;
                    break;
            }

            cnpy::NpyArray segment = cnpy::npy_load(SEGMENTS_DIR + "/" + filePath);

            for (int j = 0; j < 3; j++){
                prevEnd[j] = prevSegment[prevSegment.size() -1][j]; // prevSegment의 마지막 데이터 값
                prevEndOneBefore[j] = prevSegment[prevSegment.size() -2][j];
            }

            //segment 선택
            deliverSegment = getNextSegment_SegSeg(prevEndOneBefore, prevEnd, segment, true, true);
        
            // segment 보정 (무성구간에 따라서 값 보정)
            deliverSegment = multExpToSegment(energy, deliverSegment, 0.01, 10);

            // if (first_segment_flag == 1) {
            //     // 첫 세그먼트일 경우 이전 값들과 자연스럽게 이어지도록 보정
            //     std::vector<double> startPose = { prevEnd[0], prevEnd[1], prevEnd[2] };
            //     deliverSegment = applyOffsetDecay(startPose, deliverSegment, 9); // 5프레임 동안 보정 적용
            //     first_segment_flag = 0;
            // }
            // else {
            // // 이전 Segment의 끝부분과 현재 Segment의 시작부분을 B-spline 보간법을 통해 자연스럽게 이어줌.
            //     deliverSegment = connectTwoSegments(prevSegment, deliverSegment, 3, 5, 3);
            // }

            deliverSegment = connectTwoSegments(prevSegment, deliverSegment, 3, 5, 3);

            // 현재 세그먼트를 다음 반복을 위해 저장
            prevSegment = deliverSegment;
        }

        {
            std::lock_guard<std::mutex> lock(mouth_motion_queue_mutex);
            for ( const auto& result : motion_results) {
                mouth_motion_queue.push(std::make_pair(cycle_num, result));
            }
            head_motion_queue.push(deliverSegment);
        }
        mouth_motion_queue_cv.notify_one();

        {
            auto now = std::chrono::high_resolution_clock::now();
            std::lock_guard<std::mutex> lock(cout_mutex);
            std::cout << "Generate motion cycle " << cycle_num << " at "
                      << std::chrono::duration_cast<std::chrono::milliseconds>(now - start_time).count()
                      << " ms" << std::endl;
        }
    }
}


void control_motor(CustomSoundStream& soundStream, std::string mode_label) {
    // 모터 초기 설정 코드
    while(!mouth_motion_queue.empty()) mouth_motion_queue.pop();
    while(!head_motion_queue.empty()) head_motion_queue.pop();

    #ifdef MOTOR_ENABLED
    // GroupSyncWrite, GroupBulkRead 인스턴스 초기화
    dynamixel::GroupSyncWrite groupSyncWritePosition(portHandler, packetHandler, ADDR_PRO_GOAL_POSITION, LEN_PRO_GOAL_POSITION);
    dynamixel::GroupSyncWrite groupSyncWriteProfileVelocity(portHandler, packetHandler, ADDR_PRO_PROFILE_VELOCITY, LEN_PRO_PROFILE_VELOCITY);
    dynamixel::GroupSyncWrite groupSyncWriteGoalVelocity(portHandler, packetHandler, ADDR_PRO_GOAL_VELOCITY, LEN_PRO_GOAL_VELOCITY);
    dynamixel::GroupBulkRead groupBulkRead(portHandler, packetHandler);

    int DXL_ID[DXL_NUM] = { DXL1_ID, DXL2_ID, DXL3_ID, DXL4_ID, DXL5_ID };
    int DXL_initial_position[DXL_NUM] = { DEFAULT_PITCH, DEFAULT_ROLL_R, DEFAULT_ROLL_L, DEFAULT_YAW, DEFAULT_MOUTH };
    uint8_t dxl_error = 0;
    #else
    std::cout << "[DUMMY MOTOR] control_motor (" << mode_label << ") start." << std::endl;
    #endif

    std::vector<std::vector<double>> current_motion_data(9, std::vector<double>(3, 0.0));

    for (int cycle_num = 0;; cycle_num++) {
        if (user_interruption_flag) {
            std::cout << "Interruption detected in control_motor." << std::endl;
            break;
        }
        
        wait_for_next_cycle(cycle_num);

        std::pair<int, float> motion_data;

        std::unique_lock<std::mutex> lock(mouth_motion_queue_mutex);
        mouth_motion_queue_cv.wait(lock, [&] {
            return (stop_flag && mouth_motion_queue.empty()) || (!mouth_motion_queue.empty() && !head_motion_queue.empty());
        });
        if(!head_motion_queue.empty()){
            current_motion_data = head_motion_queue.front(); // 슬라이스 데이터 가져오기
            head_motion_queue.pop();
        }
        lock.unlock();
        
        if (stop_flag && mouth_motion_queue.empty()) {
            std::cout << "control_motor break1 -------------------- " << get_time_str() << std::endl;
            break;
        }
        int num_motor_updates = INTERVAL_MS / 40;

        std::vector<int> DXL_goal_position_vec;

        if (cycle_num == 0) {
            json led_msg;
            led_msg["cmd"] = "led_ring";
            led_msg["r"] = 50;
            led_msg["g"] = 50;
            led_msg["b"] = 233;
            webSocket.sendText(led_msg.dump());

            start_time = std::chrono::high_resolution_clock::now();
            soundStream.play(); // 첫 사이클에서 오디오 재생
            // 로그 출력
            auto playback_start_time = std::chrono::high_resolution_clock::now();
            {
                std::lock_guard<std::mutex> lock(cout_mutex);
                auto playback_start_ms = std::chrono::duration_cast<std::chrono::milliseconds>(playback_start_time.time_since_epoch()).count();
                std::cout << "[시간 측정] start → 오디오 재생 시작: "
                        << std::chrono::duration_cast<std::chrono::milliseconds>(playback_start_time - start_time).count() << "ms\n" 
                        << "[시간 측정] stt 완료 → 오디오 재생 시작: "
                        << static_cast<long long>(std::chrono::duration_cast<std::chrono::milliseconds>(playback_start_time.time_since_epoch()).count() - STT_DONE_TIME) << "ms" << std::endl;
            }
        }
        
        for (int i = 0; i < num_motor_updates; ++i) {
            //cout << "stop flag : " << stop_flag << " motion queue size : " << mouth_motion_queue.size() << '\n';
            {
                std::unique_lock<std::mutex> lock(mouth_motion_queue_mutex);
                
                if (stop_flag && mouth_motion_queue.empty()) {
                    std::cout << "motion queue size :  " << mouth_motion_queue.size() << ", control_motor (" << mode_label << ") break2 -------------------- " << get_time_str() << std::endl;
                    return;
                }
                //cout << "cycle 에 들어옴 " << '\n';
                // 현재 사이클 번호에 해당하는 모션 값이 큐에 있을 때까지 대기
                // std::cout << "mouth_motion_queue front cycle: " << mouth_motion_queue.front().first 
                //  << ", current cycle_num: " << cycle_num - 1 << '\n';

                mouth_motion_queue_cv.wait(lock, [&] {
                    return (stop_flag && mouth_motion_queue.empty()) || (!mouth_motion_queue.empty() && mouth_motion_queue.front().first == cycle_num - 1);
                });
                
                // 모션 값 가져오기
                motion_data = mouth_motion_queue.front();
                mouth_motion_queue.pop();
                
            }


            
            // 모터 제어 로직 구현
            float motor_value = motion_data.second;
            double roll_final = current_motion_data[i][0];
            double pitch_final = current_motion_data[i][1]/2;
            double yaw_final = current_motion_data[i][2];
            double mouth_final = motor_value / 60;

            DXL_goal_position_vec = RPY2DXL(roll_final, pitch_final, yaw_final, mouth_final, 0);

            #ifdef MOTOR_ENABLED
            update_DXL_goal_position(DXL_goal_position,
                                     DXL_goal_position_vec[0],
                                     DXL_goal_position_vec[1],
                                     DXL_goal_position_vec[2],
                                     DXL_goal_position_vec[3],
                                     DXL_goal_position_vec[4]);

            

            if (first_move_flag == 1) {
                for (int i = 0; i < DXL_NUM; i++) {
                    DXL_past_position[i] = DXL_goal_position[i];
                }
                first_move_flag = 0;
            } else {
                for (int i = 0; i < DXL_NUM; i++) {
                    DXL_goal_position[i] = (DXL_past_position[i] + DXL_goal_position[i]) / 2;
                }
            }


            // 현재 위치 및 속도 읽기
            readDXLPresentState(groupBulkRead, DXL_ID, DXL_present_velocity, DXL_present_position);

            // 모터를 목표 위치로 이동
            if (OPERATING_MODE == 1) {
                // 속도제어 모드
                for (int i = 0; i < DXL_NUM; i++) {
                    DXL_goal_velocity[i] = calculateDXLGoalVelocity_timeBased_ds(DXL_present_position[i], DXL_goal_position[i], DXL_present_velocity[i], GOAL_VELOCITY_CALC_TA, CONTROL_MS);
                }
                moveDXLwithVelocity(groupSyncWriteGoalVelocity, DXL_ID, DXL_goal_velocity);
            }
            else {
                // 위치제어 모드
                auto dxl_start = std::chrono::steady_clock::now();
                moveDXLtoDesiredPosition(groupSyncWriteProfileVelocity, groupSyncWritePosition, DXL_ID, DXL_goal_position, DXL_PROFILE_VELOCITY);
                auto dxl_end = std::chrono::steady_clock::now();
                auto dxl_duration = std::chrono::duration_cast<std::chrono::microseconds>(dxl_end - dxl_start);

                if (dxl_duration.count() > 1000) {
                    std::cout << "Warning: DXL processing took " << (dxl_duration.count()/1000.0) 
                            << "ms" << std::endl;
                }
            }
            
            // 이전 위치 업데이트
            for (int i = 0; i < DXL_NUM; i++) {
                DXL_past_position[i] = DXL_goal_position[i];
            }

            updatePrevValues(roll_final, pitch_final, yaw_final, mouth_final);

            double DXL_goal_rpy[4] = {roll_final, pitch_final, yaw_final, mouth_final};

            robotLogger.log(mode_label, DXL_goal_rpy, DXL_goal_position, DXL_present_position, DXL_present_velocity);

            if (i == 0 and cycle_num % 10 == 0) {
                auto expected_playback_ms = (cycle_num) * INTERVAL_MS;
                float actual_playback_ms = 0.0f;
                if (soundStream.getStatus() == sf::Sound::Playing) {
                    actual_playback_ms = soundStream.getPlayingOffset().asMilliseconds();
                }
                float playback_diff_ms = actual_playback_ms - expected_playback_ms;
                
                auto now = std::chrono::high_resolution_clock::now();
                auto motion_elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - start_time);
                auto expected_motion = cycle_num * INTERVAL_MS;

                std::cout << "Cycle " << cycle_num << ": motion_elapsed=" << motion_elapsed.count()
                            << "ms, expected=" << expected_motion
                            << "ms, diff=" << (motion_elapsed.count() - expected_motion) << "ms" << std::endl;
                std::cout << "Cycle " << cycle_num << ": playback_elapsed=" << actual_playback_ms
                            << "ms, expected=" << expected_playback_ms
                            << "ms, diff=" << playback_diff_ms << "ms" << std::endl;
            }
            #endif

            // 필요한 경우 대기 시간 추가
            // std::this_thread::sleep_for(std::chrono::milliseconds(39));
            std::this_thread::sleep_until(start_time + std::chrono::milliseconds(cycle_num * INTERVAL_MS + i * 40 + 40));
        }
    }
}

std::vector<std::string> csv_read_row(std::istream& in, char delimiter) {
    std::stringstream ss;
    bool inquotes = false;
    std::vector<std::string> row;
    while (in.good())
    {
        char c = in.get();
        if (!inquotes && c == '"') {
            inquotes = true;
        }
        else if (inquotes && c == '"') {
            if (in.peek() == '"') {
                ss << (char)in.get();
            } else {
                inquotes = false;
            }
        }
        else if (!inquotes && c == delimiter) {
            row.push_back(ss.str());
            ss.str(""); ss.clear();
        }
        else if (!inquotes && (c == '\r' || c == '\n')) {
            if (in.peek() == '\n') in.get();
            row.push_back(ss.str());
            return row;
        }
        else {
            ss << c;
        }
    }
    // 파일 끝까지 왔는데 남은 스트링이 있으면
    if (!ss.str().empty())
        row.push_back(ss.str());
    return row;
}

void wait_control_motor(){
    auto debug_0 = std::chrono::high_resolution_clock::now();
    // 모터 초기 설정 코드
    if(wait_mode_flag == false) return;
    while(!mouth_motion_queue.empty()) mouth_motion_queue.pop();
    while(!head_motion_queue.empty()) head_motion_queue.pop();

    #ifdef MOTOR_ENABLED
    // GroupSyncWrite 인스턴스 초기화
    dynamixel::GroupSyncWrite groupSyncWritePosition(portHandler, packetHandler, ADDR_PRO_GOAL_POSITION, LEN_PRO_GOAL_POSITION);
    dynamixel::GroupSyncWrite groupSyncWriteProfileVelocity(portHandler, packetHandler, ADDR_PRO_PROFILE_VELOCITY, LEN_PRO_PROFILE_VELOCITY);
    dynamixel::GroupSyncWrite groupSyncWriteGoalVelocity(portHandler, packetHandler, ADDR_PRO_GOAL_VELOCITY, LEN_PRO_GOAL_VELOCITY);
    dynamixel::GroupBulkRead groupBulkRead(portHandler, packetHandler);

    int DXL_ID[DXL_NUM] = { DXL1_ID, DXL2_ID, DXL3_ID, DXL4_ID, DXL5_ID };
    int DXL_initial_position[DXL_NUM] = { DEFAULT_PITCH, DEFAULT_ROLL_R, DEFAULT_ROLL_L, DEFAULT_YAW, DEFAULT_MOUTH };
    uint8_t dxl_error = 0;

    json led_msg;
    led_msg["cmd"] = "led_ring";
    led_msg["r"] = 233;
    led_msg["g"] = 233;
    led_msg["b"] = 50;
    webSocket.sendText(led_msg.dump());

    std::cout << "대기 모드 (wait_control_motor) 시작: " << get_time_str() << std::endl;
    #else
    // --- 가짜 모터 초기화 ---
    std::cout << "[DUMMY MOTOR] 대기 모드 (wait_control_motor) 시작." << std::endl;
    #endif

    // std::this_thread::sleep_for(std::chrono::milliseconds(200));

    auto wait_start_time = std::chrono::high_resolution_clock::now();
    int step = 0;
    constexpr auto FRAME_INTERVAL = std::chrono::milliseconds(35);

    while(wait_mode_flag == true){
        #ifdef MOTOR_ENABLED
        auto debug_1 = std::chrono::high_resolution_clock::now();
        std::ifstream headGesture(IDLE_MOTION_FILE);
        if (!headGesture) {
            std::cerr << "Empty HeadGesture File not found." << std::endl;
            return;
        }
        auto debug_2 = std::chrono::high_resolution_clock::now();

        // 초기 프레임 궤적 보간
        int SKIP_FRAMES = 20;
        std::vector<std::vector<double>> targetTraj;

        for (int i = 0; i < SKIP_FRAMES; i++) {
            if (!headGesture.good()) break;
            auto headRow = csv_read_row(headGesture, ',');
            float roll_s = std::stof(headRow[0]);
            float pitch_s = std::stof(headRow[1]);
            float yaw_s = std::stof(headRow[2]);
            float mouth_s = 0;

            double ratiooo = 2.0;

            targetTraj.push_back({roll_s * ratiooo, pitch_s * ratiooo, yaw_s * ratiooo, mouth_s});
        }

        std::vector<double> startPose;
        {
            std::lock_guard<std::mutex> lock(prev_values_mutex);
            startPose = prevValues.back();
        }

		targetTraj = applyOffsetDecay(startPose, targetTraj, SKIP_FRAMES);

        auto debug_3 = std::chrono::high_resolution_clock::now();
        std::cout << "대기모드 시작 전처리 시간: "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(debug_1 - debug_0).count() << "ms (모터 초기화), "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(debug_2 - debug_1).count() << "ms (파일 오픈), "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(debug_3 - debug_2).count() << "ms (보간)\n";
        
        // 모션 재생
        while(headGesture.good()){
            double roll_final, pitch_final, yaw_final, mouth_final;

            if (step < SKIP_FRAMES) {
                roll_final = targetTraj[step][0];
                pitch_final = targetTraj[step][1];
                yaw_final = targetTraj[step][2];
                mouth_final = targetTraj[step][3];
            } 
            else {
                auto headRow = csv_read_row(headGesture, ',');
                if(wait_mode_flag == false) break;
                double roll_s = std::stod(headRow[0]);
                double pitch_s = std::stod(headRow[1]);
                double yaw_s = std::stod(headRow[2]);
                double mouth_s = 0;

                double ratiooo = 2.0;

                roll_final = roll_s * ratiooo;
                pitch_final = pitch_s * ratiooo;
                yaw_final = yaw_s * ratiooo;
                mouth_final = mouth_s;
            }

            std::vector<int> DXL = RPY2DXL(roll_final , pitch_final, yaw_final, mouth_final, 0);
            
            update_DXL_goal_position(DXL_goal_position,
                                        DXL[0],
                                        DXL[1],
                                        DXL[2],
                                        DXL[3],
                                        DXL[4]);

            // 모터 값 읽기
            readDXLPresentState(groupBulkRead, DXL_ID, DXL_present_velocity, DXL_present_position);

            // 속도제어 모드
            if (OPERATING_MODE == 1) {
                for (int i = 0; i < DXL_NUM; i++) {
                    DXL_goal_velocity[i] = calculateDXLGoalVelocity_timeBased_ds(DXL_present_position[i], DXL_goal_position[i], DXL_present_velocity[i], GOAL_VELOCITY_CALC_TA, 35);
                }
                
                moveDXLwithVelocity(groupSyncWriteGoalVelocity, DXL_ID, DXL_goal_velocity);
            }
            // 위치제어 모드
            else {
                moveDXLtoDesiredPosition(groupSyncWriteProfileVelocity, groupSyncWritePosition, DXL_ID, DXL_goal_position, DXL_PROFILE_VELOCITY);
            }

            // 이전 위치 업데이트
            updatePrevValues(roll_final, pitch_final, yaw_final, mouth_final);
            for (int i = 0; i < DXL_NUM; i++) {
                DXL_past_position[i] = DXL_goal_position[i];
            }

            double DXL_goal_rpy[4] = {roll_final, pitch_final, yaw_final, mouth_final};

            robotLogger.log("WAIT", DXL_goal_rpy, DXL_goal_position, DXL_present_position, DXL_present_velocity);
            
            step ++;
            std::this_thread::sleep_until(wait_start_time + FRAME_INTERVAL * step);
        }
        #else
        // --- 가짜 모터 대기 동작 ---
        if(wait_mode_flag == false) break;
        // 실제 모션 파일은 읽지 않고, 대기 중임을 알리며 잠시 대기
        std::cout << "[DUMMY MOTOR] 대기 모드 동작 중..." << std::endl;
        std::this_thread::sleep_for(std::chrono::seconds(1)); // 1초마다 로그 출력
        #endif
    }
    std::cout << "wait mode finish " << std::endl;
   // portHandler -> closePort(); // 이거 계속 쓸꺼면 control_motor 함수에도 추가해주기 
}

// 모션 csv 파일 읽어서 재생하는 함수 (테스트용)
void csv_control_motor(){
    // 모터 초기 설정 코드
    if(wait_mode_flag == false) return;
    while(!mouth_motion_queue.empty()) mouth_motion_queue.pop();
    while(!head_motion_queue.empty()) head_motion_queue.pop();

    #ifdef MOTOR_ENABLED
    // GroupSyncWrite 인스턴스 초기화
    dynamixel::GroupSyncWrite groupSyncWritePosition(portHandler, packetHandler, ADDR_PRO_GOAL_POSITION, LEN_PRO_GOAL_POSITION);
    dynamixel::GroupSyncWrite groupSyncWriteProfileVelocity(portHandler, packetHandler, ADDR_PRO_PROFILE_VELOCITY, LEN_PRO_PROFILE_VELOCITY);
    dynamixel::GroupSyncWrite groupSyncWriteGoalVelocity(portHandler, packetHandler, ADDR_PRO_GOAL_VELOCITY, LEN_PRO_GOAL_VELOCITY);
    dynamixel::GroupBulkRead groupBulkRead(portHandler, packetHandler);

    int DXL_ID[DXL_NUM] = { DXL1_ID, DXL2_ID, DXL3_ID, DXL4_ID, DXL5_ID };
    int DXL_initial_position[DXL_NUM] = { DEFAULT_PITCH, DEFAULT_ROLL_R, DEFAULT_ROLL_L, DEFAULT_YAW, DEFAULT_MOUTH };
    uint8_t dxl_error = 0;

    std::ofstream csv_file;
    csv_file.open("csv_mode_test_log.csv");
    csv_file << "Timestamp(ms),Target Position,Target Velocity,Present Position,Present Velocity\n";

    std::cout << "모션 CSV 재생 모드 (csv_control_motor) 시작." << std::endl;
    #else
    // --- 가짜 모터 초기화 ---
    std::cout << "[DUMMY MOTOR] 모션 CSV 재생 모드 (csv_control_motor) 시작." << std::endl;
    #endif

    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    auto wait_start_time = std::chrono::high_resolution_clock::now();

    constexpr auto FRAME_INTERVAL = std::chrono::milliseconds(40);
    std::string line;
    while(wait_mode_flag == true){
        #ifdef MOTOR_ENABLED
        std::ifstream headGesture(HEAD_GESTURE_FILE);
        if (!headGesture) {
            std::cerr << "Empty HeadGesture File not found." << std::endl;
            return;
        }
        std::ifstream MouthGesture(MOUTH_MOTION_FILE);
        if (!MouthGesture) {
            std::cerr << "Empty MouthGesture File not found." << std::endl;
            return;
        }
        int step = 0;
        while(headGesture.good()){
            if (music.getStatus() != sf::Music::Playing) {
                music.play();
            }
            auto headRow = csv_read_row(headGesture, ',');
            auto mouthRow = csv_read_row(MouthGesture, ',');
            if(wait_mode_flag == false) break;
            float roll_s = std::stof(headRow[0]);
            float pitch_s = std::stof(headRow[1]);
            float yaw_s = std::stof(headRow[2]);
            float mouth_s = std::stof(mouthRow[0]);

            // double ratiooo = 1.4;
            float ratiooo = std::stof(mouthRow[1]) * 2.0;

            std::vector<int> DXL = RPY2DXL(roll_s * ratiooo , pitch_s *ratiooo, yaw_s * ratiooo, mouth_s / 60.0, 0);
            
            update_DXL_goal_position(DXL_goal_position,
                                        DXL[0],
                                        DXL[1],
                                        DXL[2],
                                        DXL[3],
                                        DXL[4]);

            // 모터 제어
            // 속도제어 모드
            if (OPERATING_MODE == 1) {
                readDXLPresentState(groupBulkRead, DXL_ID, DXL_present_velocity, DXL_present_position);
                for (int i = 0; i < DXL_NUM; i++) {
                    DXL_goal_velocity[i] = calculateDXLGoalVelocity_timeBased_ds(DXL_present_position[i], DXL_goal_position[i], DXL_present_velocity[i], GOAL_VELOCITY_CALC_TA, 35);
                }
                
                moveDXLwithVelocity(groupSyncWriteGoalVelocity, DXL_ID, DXL_goal_velocity);
            }
            // 위치제어 모드
            else {
                moveDXLtoDesiredPosition(groupSyncWriteProfileVelocity, groupSyncWritePosition, DXL_ID, DXL_goal_position, DXL_PROFILE_VELOCITY);
            }

            // 이전 위치 업데이트
            updatePrevValues(roll_s * ratiooo , pitch_s *ratiooo, yaw_s * ratiooo, mouth_s);
            for (int i = 0; i < DXL_NUM; i++) {
                DXL_past_position[i] = DXL_goal_position[i];
            }

            auto current_time = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::milli> timestamp = current_time - wait_start_time;
            csv_file << timestamp.count() << ","
                    << DXL_goal_position[0] << ","
                    << DXL_goal_velocity[0] << ","
                    << DXL_present_position[0] << ","
                    << DXL_present_velocity[0] << "\n";
            
            // 동작과 소리 싱크 확인
            sf::Int32 music_ms = music.getPlayingOffset().asMilliseconds();
            std::cout << "Step: " << step << ", Motor ms: " << timestamp.count() << ", Music ms: " << music_ms << ", Diff: " << (timestamp.count() - music_ms) << "ms" << std::endl;
            std::this_thread::sleep_until(wait_start_time + FRAME_INTERVAL * step);
            step ++;
        }
        #else
        // --- 모터 비활성화됨 ---
        std::cout << "[DUMMY MOTOR] 모션 CSV 재생 모드 (csv_control_motor)." << std::endl;
        #endif
    }
}


// MPU6050 초기화
void mpu6050_init() {
    wiringPiI2CWriteReg8(fd, 0x6B, 0);
}

// 16비트 데이터 읽기
int read_raw_data(int addr) {
    int high = wiringPiI2CReadReg8(fd, addr);
    int low = wiringPiI2CReadReg8(fd, addr + 1);
    int value = (high << 8) | low;

    if (value > 32768)
        value -= 65536;
    return value;
}

void initialize_robot_posture() {
    dynamixel::GroupSyncWrite groupSyncWritePosition(portHandler, packetHandler, ADDR_PRO_GOAL_POSITION, LEN_PRO_GOAL_POSITION);
    dynamixel::GroupSyncWrite groupSyncWriteVelocity(portHandler, packetHandler, ADDR_PRO_PROFILE_VELOCITY, LEN_PRO_PROFILE_VELOCITY);
    dynamixel::GroupSyncRead groupSyncReadPresentPosition(portHandler, packetHandler, ADDR_PRO_PRESENT_POSITION, LEN_PRO_PRESENT_POSITION);

    int DXL_ID[DXL_NUM] = { DXL1_ID, DXL2_ID, DXL3_ID, DXL4_ID, DXL5_ID };
        
    //현재 모터 위치를 순차적 읽기
    readDXLPresentPosition(groupSyncReadPresentPosition, DXL_ID, DXL_present_position);

    // 6) MPU6050 초기화
    if (wiringPiSetup() == -1) {
        std::cerr << "WiringPi 초기화 실패!" << std::endl;
        return;
    }
    fd = wiringPiI2CSetup(MPU6050_ADDR);
    if (fd == -1) {
        std::cerr << "MPU6050 I2C 연결 실패!" << std::endl;
        return;
    }
    mpu6050_init();
    std::cout << "MPU6050 데이터 수집 시작..." << std::endl;

    std::vector<int> DXL_goal_position;
    int Roll_L_adjust_flag = 0;
    int Roll_R_adjust_flag = 0;
    int Pitch_adjust_flag = 0;
    int mouth_adjust_flag = 0;

    const float current_threshold_mA = -20;   // 목표 전류 임계값 (mA)
    const int adjustment_increment = 20;       // 모터 위치 조정 증분 (펄스)
    bool tension_satisfied = false;
    const int sample_count = 3;

    std::cout << "Roll 조정" << std::endl;

    int sum_accel_x = 0, sum_accel_y = 0, sum_accel_z = 0;
    for (int i = 0; i < sample_count; i++) {
        sum_accel_x += read_raw_data(0x3B);
        sum_accel_y += read_raw_data(0x3D);
        sum_accel_z += read_raw_data(0x3F);
        delay(10);  // 각 샘플 사이에 짧은 딜레이
    }
    int avg_accel_x = sum_accel_x / sample_count;
    int avg_accel_y = sum_accel_y / sample_count;
    int avg_accel_z = sum_accel_z / sample_count;
    
    // 5-2. 평균 센서값을 g 단위로 변환
    float Ax = avg_accel_x / 16384.0;
    float Ay = avg_accel_y / 16384.0;
    float Az = avg_accel_z / 16384.0;

    std::cout << "AX : " << Ax << " , Ay : " << Ay << " , Az : " << Az << '\n';
    DXL_goal_position = {DEFAULT_PITCH , DEFAULT_ROLL_R, DEFAULT_ROLL_L, DEFAULT_YAW, DEFAULT_MOUTH};
    if (Ax > 0){
        // Roll_L 조정
        while(true){
            DXL_goal_position[2] -= adjustment_increment;
            moveDXLtoDesiredPosition(groupSyncWriteVelocity, groupSyncWritePosition, DXL_ID, DXL_goal_position.data(), DXL_PROFILE_VELOCITY);

            sum_accel_x = 0, sum_accel_y = 0, sum_accel_z = 0;
            for (int i = 0; i < sample_count; i++) {
                sum_accel_x += read_raw_data(0x3B);
                sum_accel_y += read_raw_data(0x3D);
                sum_accel_z += read_raw_data(0x3F);
                delay(10);  // 각 샘플 사이에 짧은 딜레이
            }
            avg_accel_x = sum_accel_x / sample_count;
            avg_accel_y = sum_accel_y / sample_count;
            avg_accel_z = sum_accel_z / sample_count;
            Ax = avg_accel_x / 16384.0;
            Ay = avg_accel_y / 16384.0;
            Az = avg_accel_z / 16384.0;

            std::cout << "AX : " << Ax << " , Ay : " << Ay << " , Az : " << Az << '\n';
            if(Ax < -0.15) break;
        }
        
        // Roll_R 조정
        while(true){
            DXL_goal_position[1] -= adjustment_increment;
            moveDXLtoDesiredPosition(groupSyncWriteVelocity, groupSyncWritePosition, DXL_ID, DXL_goal_position.data(), DXL_PROFILE_VELOCITY);

            sum_accel_x = 0, sum_accel_y = 0, sum_accel_z = 0;
            for (int i = 0; i < sample_count; i++) {
                sum_accel_x += read_raw_data(0x3B);
                sum_accel_y += read_raw_data(0x3D);
                sum_accel_z += read_raw_data(0x3F);
                delay(10);  // 각 샘플 사이에 짧은 딜레이
            }
            avg_accel_x = sum_accel_x / sample_count;
            avg_accel_y = sum_accel_y / sample_count;
            avg_accel_z = sum_accel_z / sample_count;
            Ax = avg_accel_x / 16384.0;
            Ay = avg_accel_y / 16384.0;
            Az = avg_accel_z / 16384.0;

            std::cout << "AX : " << Ax << " , Ay : " << Ay << " , Az : " << Az << '\n';
            if(Ax > 0.01) break;
        }
    }
    else if (Ax <= 0){
        while(true){
            DXL_goal_position[1] -= adjustment_increment;
            moveDXLtoDesiredPosition(groupSyncWriteVelocity, groupSyncWritePosition, DXL_ID, DXL_goal_position.data(), DXL_PROFILE_VELOCITY);

            sum_accel_x = 0, sum_accel_y = 0, sum_accel_z = 0;
            for (int i = 0; i < sample_count; i++) {
                sum_accel_x += read_raw_data(0x3B);
                sum_accel_y += read_raw_data(0x3D);
                sum_accel_z += read_raw_data(0x3F);
                delay(10);  // 각 샘플 사이에 짧은 딜레이
            }
            avg_accel_x = sum_accel_x / sample_count;
            avg_accel_y = sum_accel_y / sample_count;
            avg_accel_z = sum_accel_z / sample_count;
            Ax = avg_accel_x / 16384.0;
            Ay = avg_accel_y / 16384.0;
            Az = avg_accel_z / 16384.0;

            std::cout << "AX : " << Ax << " , Ay : " << Ay << " , Az : " << Az << '\n';
            if(Ax > 0.15) break;
        }

        // Roll_L 조정
        while(true){
            DXL_goal_position[2] -= adjustment_increment;
            moveDXLtoDesiredPosition(groupSyncWriteVelocity, groupSyncWritePosition, DXL_ID, DXL_goal_position.data(), DXL_PROFILE_VELOCITY);

            sum_accel_x = 0, sum_accel_y = 0, sum_accel_z = 0;
            for (int i = 0; i < sample_count; i++) {
                sum_accel_x += read_raw_data(0x3B);
                sum_accel_y += read_raw_data(0x3D);
                sum_accel_z += read_raw_data(0x3F);
                delay(10);  // 각 샘플 사이에 짧은 딜레이
            }
            avg_accel_x = sum_accel_x / sample_count;
            avg_accel_y = sum_accel_y / sample_count;
            avg_accel_z = sum_accel_z / sample_count;
            Ax = avg_accel_x / 16384.0;
            Ay = avg_accel_y / 16384.0;
            Az = avg_accel_z / 16384.0;

            std::cout << "AX : " << Ax << " , Ay : " << Ay << " , Az : " << Az << '\n';
            if(Ax < -0.01) break;
        }
    }
    
    std::cout << "Pitch 조정" << std::endl;

    sum_accel_x = 0, sum_accel_y = 0, sum_accel_z = 0;
    for (int i = 0; i < sample_count; i++) {
        sum_accel_x += read_raw_data(0x3B);
        sum_accel_y += read_raw_data(0x3D);
        sum_accel_z += read_raw_data(0x3F);
        delay(10);  // 각 샘플 사이에 짧은 딜레이
    }
    avg_accel_x = sum_accel_x / sample_count;
    avg_accel_y = sum_accel_y / sample_count;
    avg_accel_z = sum_accel_z / sample_count;
    Ax = avg_accel_x / 16384.0;
    Ay = avg_accel_y / 16384.0;
    Az = avg_accel_z / 16384.0;
    //pitch 조정 -일 때 생각해서 예외 처리 실행해야할 듯 
    if(Ay < 0.009){
        std::cout << "Ay < 0.009" << std::endl;
        while(true){
            DXL_goal_position[0] -= adjustment_increment;
            moveDXLtoDesiredPosition(groupSyncWriteVelocity, groupSyncWritePosition, DXL_ID, DXL_goal_position.data(), DXL_PROFILE_VELOCITY);

            sum_accel_x = 0, sum_accel_y = 0, sum_accel_z = 0;
            for (int i = 0; i < sample_count; i++) {
                sum_accel_x += read_raw_data(0x3B);
                sum_accel_y += read_raw_data(0x3D);
                sum_accel_z += read_raw_data(0x3F);
                delay(10);  // 각 샘플 사이에 짧은 딜레이
            }
            avg_accel_x = sum_accel_x / sample_count;
            avg_accel_y = sum_accel_y / sample_count;
            avg_accel_z = sum_accel_z / sample_count;
            Ax = avg_accel_x / 16384.0;
            Ay = avg_accel_y / 16384.0;
            Az = avg_accel_z / 16384.0;

            std::cout << "AX : " << Ax << " , Ay : " << Ay << " , Az : " << Az << '\n';
            if(Ay > 0.009) break;
        }
    }
    else{
        std::cout << "Ay > 0.009" << std::endl;
        //pitch가 이미 앞으로 당겨져 있을 경우 예외 처리
        int now_Ay = Ay;
        while(true){
            DXL_goal_position[0] -= adjustment_increment;
            moveDXLtoDesiredPosition(groupSyncWriteVelocity, groupSyncWritePosition, DXL_ID, DXL_goal_position.data(), DXL_PROFILE_VELOCITY);

            sum_accel_x = 0, sum_accel_y = 0, sum_accel_z = 0;
            for (int i = 0; i < sample_count; i++) {
                sum_accel_x += read_raw_data(0x3B);
                sum_accel_y += read_raw_data(0x3D);
                sum_accel_z += read_raw_data(0x3F);
                delay(10);  // 각 샘플 사이에 짧은 딜레이
            }
            avg_accel_x = sum_accel_x / sample_count;
            avg_accel_y = sum_accel_y / sample_count;
            avg_accel_z = sum_accel_z / sample_count;
            Ax = avg_accel_x / 16384.0;
            Ay = avg_accel_y / 16384.0;
            Az = avg_accel_z / 16384.0;

            std::cout << "AX : " << Ax << " , Ay : " << Ay << " , Az : " << Az << '\n';
            if(Ay > now_Ay + 0.01) break;
        }

        while(true){
            DXL_goal_position[1] -= adjustment_increment;
            DXL_goal_position[2] -= adjustment_increment;
            moveDXLtoDesiredPosition(groupSyncWriteVelocity, groupSyncWritePosition, DXL_ID, DXL_goal_position.data(), DXL_PROFILE_VELOCITY);

            sum_accel_x = 0, sum_accel_y = 0, sum_accel_z = 0;
            for (int i = 0; i < sample_count; i++) {
                sum_accel_x += read_raw_data(0x3B);
                sum_accel_y += read_raw_data(0x3D);
                sum_accel_z += read_raw_data(0x3F);
                delay(10);  // 각 샘플 사이에 짧은 딜레이
            }
            avg_accel_x = sum_accel_x / sample_count;
            avg_accel_y = sum_accel_y / sample_count;
            avg_accel_z = sum_accel_z / sample_count;
            Ax = avg_accel_x / 16384.0;
            Ay = avg_accel_y / 16384.0;
            Az = avg_accel_z / 16384.0;

            std::cout << "AX : " << Ax << " , Ay : " << Ay << " , Az : " << Az << '\n';
            if(Ay < 0.009) break;
        }
    }
    
    std::cout << "Mouth 조정" << std::endl;
    
    while(!mouth_adjust_flag){
        DXL_goal_position[4] -= adjustment_increment;
        moveDXLtoDesiredPosition(groupSyncWriteVelocity, groupSyncWritePosition, DXL_ID, DXL_goal_position.data(), DXL_PROFILE_VELOCITY);

        int id = 5;
        uint8_t dxl_err = 0;
        int dxl_comm_result = COMM_TX_FAIL;
        int16_t present_current = 0; // signed 16-bit
        int16_t sum_current = 0;
        for(int i = 0; i< 5; i++){
                dxl_comm_result = packetHandler->read2ByteTxRx(
                portHandler,
                id,
                ADDR_PRO_PRESENT_CURRENT,
                (uint16_t*)&present_current,
                &dxl_err
            );
            if (dxl_comm_result != COMM_SUCCESS) {
                std::cout << "[ID:" << id << "] read2ByteTxRx(Current) 실패: "
                        << packetHandler->getTxRxResult(dxl_comm_result) << std::endl;
            } else if (dxl_err != 0) {
                std::cout << "[ID:" << id << "] 패킷 에러: "
                        << packetHandler->getRxPacketError(dxl_err) << std::endl;
            } else {
                // XM430 기준 1 LSB ≈ 2.69 mA
                // ex) present_current = -50 -> 약 -134.5 mA
                std::cout << "[ID:" << id << "] Present Current (LSB) = "
                        << present_current << " => 약 "
                        << present_current * 2.69 << " mA" << std::endl;
             }
            sum_current += present_current;
            delay(20);
        }
        sum_current /= 5;

        // 텐션 판별 예시
        if (sum_current < 0) {
            std::cout << " → 음수 전류: 반대 방향으로 텐션이 걸려 있음" << std::endl;
        } else if (sum_current > 0) {
            std::cout << " → 양수 전류: 해당 방향 텐션" << std::endl;
        } else {
            std::cout << " → 전류=0, 텐션이 없다" << std::endl;
        }

        
        if (present_current < 0) mouth_adjust_flag = 1;
            else mouth_adjust_flag = 0;
            
    }
    
    finish_adjust_ready = true;
}

void cleanup_dynamixel() {
    #ifdef MOTOR_ENABLED
    std::cout << "\n모터 토크를 끄고 포트를 닫습니다..." << std::endl;
    uint8_t dxl_error = 0;
    int DXL_ID[DXL_NUM] = { DXL1_ID, DXL2_ID, DXL3_ID, DXL4_ID, DXL5_ID };
    disable_torque(packetHandler, portHandler, DXL_ID, dxl_error);
    portHandler->closePort();
    std::cout << "모터가 안전하게 종료되었습니다." << std::endl;
    #endif
}

void signal_handler(int signum) {
    std::cout << "종료 신호 (" << signum << ") 수신. 프로그램을 정리합니다." << std::endl;
    cleanup_dynamixel();
    exit(signum);
}

void robot_main_loop(std::future<void> server_ready_future) {
    std::cout << "서버 연결 대기 중..." << std::endl;
    server_ready_future.get(); // 서버가 준비될 때까지 대기
    std::cout << "서버 연결 완료!" << std::endl;

    robotLogger.createNewLogDir();
	std::thread wait_mode_thread;

    std::pair<std::string,std::string> play_music;
    while (true) {
        // --- 루프 시작 시 상태 초기화 ---
        stop_flag = false;
        is_realtime_streaming = false;
        is_responses_streaming = false;
        {
            std::lock_guard<std::mutex> lock(realtime_stream_buffer_mutex);
            realtime_stream_buffer.clear();
        }
        {
            std::lock_guard<std::mutex> lock(responses_stream_buffer_mutex);
            responses_stream_buffer.clear();
        }
        
        SF_INFO sfinfo;
        SNDFILE* sndfile = nullptr;
        bool is_file_based = false;
        bool responses_only_flag = false;
        robotLogger.startSession();
        std::string current_mode_label = "UNKNOWN";

        // --- 1. 다음 행동 결정 ---
        if(music_flag) {
            music_flag = 0;
            current_mode_label = "MUSIC";
            std::cout << "music_flag IN" << std::endl;
            std::string play_song_path = MUSIC_DIR + "/" + play_music.first + "_" + play_music.second + ".wav";
            vocal_file_path = VOCAL_DIR + "/" + play_music.first + "_" + play_music.second + "_" + "vocals" + ".wav";
            sndfile = sf_open(play_song_path.c_str(), SFM_READ, &sfinfo);
            if (sndfile) is_file_based = true;
            playing_music_flag = true;
        }
        else {
            if (!wait_mode_thread.joinable()) {
                wait_mode_flag = true;
				wait_mode_thread = std::thread(wait_control_motor);
            }
            
            json response;
            {
                std::unique_lock<std::mutex> lock(server_message_queue_mutex);
                server_message_queue_cv.wait(lock, [] { return !server_message_queue.empty(); });
                response = server_message_queue.front();
                server_message_queue.pop();
            }
            
            std::string type = response.value("type", "error");
            
            if (type == "play_audio") {
                current_mode_label = "PLAY_AUDIO";
                std::string file_to_play = response.value("file_to_play", "");
                sndfile = sf_open(file_to_play.c_str(), SFM_READ, &sfinfo);
                if (sndfile) is_file_based = true;
                if (file_to_play.find("sleep") != std::string::npos) {
                    robotLogger.createNewLogDir();
                }
            }
            else if (type == "play_music") {
                current_mode_label = "PLAY_MUSIC";
                music_flag = 1;
                std::string file_to_play = response.value("file_to_play", "");
                play_music = {response.value("title", ""), response.value("artist", "")};
                sndfile = sf_open(file_to_play.c_str(), SFM_READ, &sfinfo);
                if (sndfile) is_file_based = true;
            }
            else if (type == "realtime_stream_start") {
                is_file_based = false;
                is_realtime_streaming = true;
                current_mode_label = "REALTIME"; // 실시간 스트리밍 모드
                sfinfo.channels = AUDIO_CHANNELS;
                sfinfo.samplerate = AUDIO_SAMPLE_RATE;
            }

            else if (type == "responses_only") {
                is_file_based = false;
                is_realtime_streaming = false;
                is_responses_streaming = true;
                responses_only_flag = true;
                current_mode_label = "RESPONSE"; // 응답 모드
                sfinfo.channels = AUDIO_CHANNELS;
                sfinfo.samplerate = AUDIO_SAMPLE_RATE;
            }
        }

        if (!is_file_based && !is_realtime_streaming && !responses_only_flag) {
            std::cerr << "Error: No valid audio source." << std::endl;
            if (sndfile) sf_close(sndfile);
            continue;
        }

        CustomSoundStream soundStream(sfinfo.channels, sfinfo.samplerate);
        CustomSoundStream soundStream_resp(sfinfo.channels, sfinfo.samplerate); // Responses용 사운드 스트림

        // --- 2. 스레드 시작 ---
        is_speaking = true;
        
        if (is_file_based) {
			wait_mode_flag = false;
			if (wait_mode_thread.joinable()) {
				wait_mode_thread.join();
			}
            start_time = std::chrono::high_resolution_clock::now();
            std::thread t1(read_and_split, sndfile, sfinfo, std::ref(soundStream));
            std::thread t2(generate_motion, sfinfo.channels, sfinfo.samplerate);
            std::thread t3(control_motor, std::ref(soundStream), current_mode_label);
            t1.join();
            t2.join();
            t3.join();
        } else { // realtime or responses
            const size_t bytes_per_interval = sfinfo.samplerate * sfinfo.channels * sizeof(sf::Int16) * INTERVAL_MS / 1000;

			// Realtime 처리
            if (!responses_only_flag) {
                // 데이터가 들어올 때까지 대기
                {
                    std::unique_lock<std::mutex> lock(realtime_stream_buffer_mutex);
					// 버퍼 사이즈가 한 사이클(17280 bytes) 이상 차거나, 전체 응답이 한 사이클 분량보다 짧거나, 사용자 끼어들기 신호가 있을 경우 해제
                    realtime_stream_buffer_cv.wait(lock, [&]{ return realtime_stream_buffer.size() >= bytes_per_interval || (!is_realtime_streaming && !realtime_stream_buffer.empty()) || user_interruption_flag; });
                }

                if (!realtime_stream_buffer.empty() && !user_interruption_flag) {
                    wait_mode_flag = false;
                    if (wait_mode_thread.joinable()) {
                        wait_mode_thread.join();
                    }
                    start_time = std::chrono::high_resolution_clock::now();
                    std::thread t1_realtime(stream_and_split, std::ref(sfinfo), std::ref(soundStream), "realtime");
                    std::thread t2_realtime(generate_motion, sfinfo.channels, sfinfo.samplerate);
                    std::thread t3_realtime(control_motor, std::ref(soundStream), "REALTIME");
                    
                    t1_realtime.join();
                    t2_realtime.join();
                    t3_realtime.join();
                }
            }

            // Response 전 중간 대기
			if (!wait_mode_thread.joinable()) {
                wait_mode_flag = true;
				wait_mode_thread = std::thread(wait_control_motor);
                std::this_thread::sleep_for(std::chrono::milliseconds(500));
			}
            
            // Responses 처리
            if (!user_interruption_flag) {
                // std::this_thread::sleep_for(std::chrono::milliseconds(500));
                
                // 스레드 큐 초기화
                {
                    std::lock_guard<std::mutex> lock(audio_queue_mutex);
                    std::queue<std::vector<float>> empty_audio_q;
                    std::swap(audio_queue, empty_audio_q);
                }
                
                {
                    std::lock_guard<std::mutex> lock(mouth_motion_queue_mutex);
                    std::queue<std::pair<int, float>> empty_mouth_q;
                    std::swap(mouth_motion_queue, empty_mouth_q);
                    std::queue<std::vector<std::vector<double>>> empty_head_q;
                    std::swap(head_motion_queue, empty_head_q);
                }

                // Responses 스트림이 시작되고 데이터가 들어올 때까지 대기
                {
                    std::unique_lock<std::mutex> lock(responses_stream_buffer_mutex);
                    responses_stream_buffer_cv.wait(lock, [&]{ return responses_stream_buffer.size() >= bytes_per_interval || (!is_responses_streaming && !responses_stream_buffer.empty()) || user_interruption_flag; });
                }

                if (!responses_stream_buffer.empty() && !user_interruption_flag) {
                    wait_mode_flag = false;
					if (wait_mode_thread.joinable()) {
						wait_mode_thread.join();
					}

                    stop_flag = false; // 다음 재생을 위해 stop_flag 리셋
                    start_time = std::chrono::high_resolution_clock::now();
                    std::thread t1_responses(stream_and_split, std::ref(sfinfo), std::ref(soundStream_resp), "responses");
                    std::thread t2_responses(generate_motion, sfinfo.channels, sfinfo.samplerate);
                    std::thread t3_responses(control_motor, std::ref(soundStream_resp), "RESPONSES");

                    t1_responses.join();
                    t2_responses.join();
                    t3_responses.join();
                }
            }
        }
        
        is_speaking = false;

        if (user_interruption_flag) {
            std::cout << "Interruption handling: Cleaning up resources." << std::endl;
            soundStream.stop();
            soundStream.clearBuffer();
            
            if (is_responses_streaming || !responses_stream_buffer.empty()) {
                soundStream_resp.stop();
                soundStream_resp.clearBuffer();
            }
            
            std::queue<std::vector<float>> empty_audio_q;
            std::swap(audio_queue, empty_audio_q);
            std::queue<std::pair<int, float>> empty_mouth_q;
            std::swap(mouth_motion_queue, empty_mouth_q);
            std::queue<std::vector<std::vector<double>>> empty_head_q;
            std::swap(head_motion_queue, empty_head_q);

            if (sndfile) sf_close(sndfile);
            playing_music_flag = false;
            continue; // 메인 루프의 처음으로 돌아가 다음 명령을 기다림
        }


        // 오디오 재생이 끝날 때까지 대기
        // while (soundStream.getStatus() == sf::Sound::Playing) {
        //     std::this_thread::sleep_for(std::chrono::milliseconds(10));
        // }
        // if (is_responses_streaming || !responses_stream_buffer.empty()) {
        //     while (soundStream_resp.getStatus() == sf::Sound::Playing) {
        //         std::this_thread::sleep_for(std::chrono::milliseconds(10));
        //     }
        // }

        robotLogger.closeSession();
        if (sndfile) sf_close(sndfile);
        playing_music_flag = false;
    }
}

int main() {
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);

    #ifdef MOTOR_ENABLED
    if (!initialize_dynamixel()) {
        std::cerr << "모터 초기화 실패!" << std::endl;
        return -1;
    }

    // 초기 자세로 이동
    if (OPERATING_MODE == 1)
        move_to_initial_position_velctrl();
    else
        move_to_initial_position_posctrl();
    std::this_thread::sleep_for(std::chrono::seconds(1));
    // 자이로센서를 이용한 로봇 초기자세 설정
    // initialize_robot_posture();
    #endif


    // 임시 오디오 재생 코드
    std::string audioFilePath = "assets/audio/가까운 듯 먼 그대여.mp3"; // 재생할 오디오 파일 경로
    if (!music.openFromFile(audioFilePath)) {
        std::cerr << "Error: Could not load audio file: " << audioFilePath << std::endl;
        return -1;
    }
    std::cout << "Audio: File loaded. Duration: " << music.getDuration().asSeconds() << " seconds" << std::endl;
    float volume = 100.0f; // 볼륨 설정 (0.0f ~ 100.0f)
    music.setVolume(volume);
    std::cout << "Audio: Ready! Waiting for start signal..." << std::endl;
    // ===== 임시 끝 =====


    // 웹소켓 서버 준비
    std::future<void> server_ready_future = server_ready_promise.get_future();
    ix::initNetSystem();
    webSocket.setUrl("ws://127.0.0.1:5000");
    webSocket.setOnMessageCallback([](const ix::WebSocketMessagePtr& msg) {
        if (msg->type == ix::WebSocketMessageType::Message) {
            try {
                json response = json::parse(msg->str);
                std::string type = response.value("type", "");

                if (type == "realtime_audio_chunk") {
                    if (user_interruption_flag) return;
                    std::string b64_data = response.value("data", "");
                    std::string decoded_data;
                    macaron::Base64::Decode(b64_data, decoded_data);
                    std::lock_guard<std::mutex> lock(realtime_stream_buffer_mutex);
                    realtime_stream_buffer.insert(realtime_stream_buffer.end(), decoded_data.begin(), decoded_data.end());
                    realtime_stream_buffer_cv.notify_one();
                } else if (type == "realtime_stream_end") {
                    is_realtime_streaming = false;
                    realtime_stream_buffer_cv.notify_one();
                } else if (type == "responses_audio_chunk") {
                    if (user_interruption_flag) return;
                    std::string b64_data = response.value("data", "");
                    std::string decoded_data;
                    macaron::Base64::Decode(b64_data, decoded_data);
                    std::lock_guard<std::mutex> lock(responses_stream_buffer_mutex);
                    responses_stream_buffer.insert(responses_stream_buffer.end(), decoded_data.begin(), decoded_data.end());
                    responses_stream_buffer_cv.notify_one();
                } else if (type == "responses_stream_start") {
                    is_responses_streaming = true;
                    responses_stream_buffer_cv.notify_one();
                } else if (type == "responses_stream_end") {
                    is_responses_streaming = false;
                    responses_stream_buffer_cv.notify_one();
                } else if (type == "stt_done") {
                    if (response.contains("stt_done_time")) {
                        STT_DONE_TIME = response["stt_done_time"].get<double>();
                    }
                } else if (type == "user_interruption") {
                    if (is_speaking) {
                        std::cout << "[WebSocket] User interruption received." << std::endl;
                        user_interruption_flag = true;
                        realtime_stream_buffer_cv.notify_all();
                        responses_stream_buffer_cv.notify_all();
                        audio_queue_cv.notify_all();
                        mouth_motion_queue_cv.notify_all();
                    }
                } else { // audio_chunk가 아닌 다른 모든 메시지(gpt_streaming_start, play_audio 등)는 메인 루프가 처리하도록 큐에 넣음
                    // 새로운 재생 시작을 알리는 모든 메시지 유형에 대해 인터럽트 플래그를 즉시 리셋
                    if (type == "realtime_stream_start" || type == "play_audio" || type == "play_music" || type == "responses_only") {
                        user_interruption_flag = false;
                    }
                    std::lock_guard<std::mutex> lock(server_message_queue_mutex);
                    server_message_queue.push(response);
                    server_message_queue_cv.notify_one();
                }
            } catch (const json::parse_error& e) {
                std::cerr << "JSON 파싱 오류: " << e.what() << " | 원본 메시지: " << msg->str << std::endl;
            }
        } else if (msg->type == ix::WebSocketMessageType::Open) {
            std::cout << "[WebSocket] 서버에 성공적으로 연결되었습니다." << std::endl;
            server_ready_promise.set_value();
        } else if (msg->type == ix::WebSocketMessageType::Error) {
            std::cerr << "[WebSocket] 연결 오류: " << msg->errorInfo.reason << std::endl;
        }
    });

    // 웹소켓 서버 및 메인 루프 시작
    webSocket.start();
    std::thread robot_thread(robot_main_loop, std::move(server_ready_future));
    robot_thread.join();
    webSocket.stop();
    ix::uninitNetSystem();
    cleanup_dynamixel();
    return 0;
}