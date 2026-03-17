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
#ifndef _WIN32
#include <unistd.h>
#endif
#include <csignal>
#include "cnpy.h"
#include "Macro_function.h"
#include "MotionLogger.h"
#include "Config.h"

// 하드웨어 전용 헤더 (MOTOR_ENABLED 시에만 포함)
#ifdef MOTOR_ENABLED
#include <wiringPiI2C.h>
#include <wiringPi.h>
#include "DynamixelDriver.h"
#endif

// WebSocket 및 JSON 관련 헤더
#include <ixwebsocket/IXNetSystem.h>
#include <ixwebsocket/IXWebSocket.h>
#include <ixwebsocket/IXWebSocketServer.h>
#include <ixwebsocket/IXBase64.h>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

// MOTOR_ENABLED는 CMake에서 제어 (cmake -DMOTOR_ENABLED=OFF ..)

static constexpr int INTERVAL_MS = 360; // 시퀀스 1개 당 시간
static constexpr int CONTROL_MS = 40; // 모터 제어 주기
static constexpr int AUDIO_SAMPLE_RATE = 24000;
static constexpr int AUDIO_CHANNELS = 1;

// 파일 경로 설정
const std::string ASSETS_DIR = "assets";
const std::string DATA_DIR = "data";
const std::string MUSIC_DIR = ASSETS_DIR + "/audio/music";
const std::string VOCAL_DIR = ASSETS_DIR + "/audio/vocal";
const std::string SEGMENTS_DIR = DATA_DIR + "/segments";
const std::string IDLE_MOTION_FILE = DATA_DIR + "/empty_10min.csv";

// 전역 변수 및 동기화 도구
std::string vocal_file_path;

std::chrono::time_point<std::chrono::high_resolution_clock> start_time; // 쓰레드 대기 시간 설정용

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



#ifdef MOTOR_ENABLED
DynamixelDriver* dxl_driver = nullptr;
DataLogger motion_logger;
HighFreqLogger* tuning_logger = nullptr;
#endif


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

// WebSocket 서버 관련 전역 객체
ix::WebSocketServer* g_ws_server = nullptr;
ix::WebSocket* g_client_ws = nullptr;  // 연결된 Python 클라이언트
std::mutex g_client_ws_mutex;
std::queue<json> server_message_queue;
std::mutex server_message_queue_mutex;
std::condition_variable server_message_queue_cv;
std::promise<void> server_ready_promise;
std::atomic<bool> server_ready_fired{false};

// 스트리밍 데이터 처리를 위한 전역 변수
std::atomic<bool> is_responses_streaming(false);
std::vector<uint8_t> responses_stream_buffer;
std::mutex responses_stream_buffer_mutex;
std::condition_variable responses_stream_buffer_cv;

// Python 클라이언트에 JSON 메시지 전송 (스레드 안전)
void send_to_python(const json& msg) {
    std::lock_guard<std::mutex> lock(g_client_ws_mutex);
    if (g_client_ws) {
        g_client_ws->sendText(msg.dump());
    }
}

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

// CSV파일 행 읽기 함수
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

// CustomSoundStream 클래스 정의
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

// Idle Motion 관리 클래스
// 대기-말하기 간 동일한 Headmotion csv파일을 참조할 때 연속성을 위해 구현.
class IdleMotionManager {
public:
    struct Pose {
        double r, p, y;
    };

    static IdleMotionManager& getInstance() {
        static IdleMotionManager instance;
        return instance;
    }

    bool loadCSV(const std::string& filepath) {
        std::ifstream file(filepath);
        if (!file.is_open()) {
            std::cerr << "Failed to open idle motion file: " << filepath << std::endl;
            return false;
        }

        frames.clear();
        while (file.good()) {
            auto row = csv_read_row(file, ',');
            if (row.size() < 3) continue;
            try {
                double r = std::stod(row[0]);
                double p = std::stod(row[1]);
                double y = std::stod(row[2]);
                frames.push_back({r, p, y});
            } catch (...) { continue; }
        }
        std::cout << "Idle motions loaded: " << frames.size() << " frames." << std::endl;
        return !frames.empty();
    }

    // 다음 프레임 데이터를 가져오고 인덱스 증가
    Pose getNextPose(double ratio = 1.0) {
        if (frames.empty()) return {0, 0, 0};
        
        // 현재 인덱스의 데이터 반환
        Pose p = frames[currentIndex];

        // 인덱스 증가 및 순환 (Loop)
        currentIndex = (currentIndex + 1) % frames.size();
        return {p.r * ratio, p.p * ratio, p.y * ratio};
    }

    // N개의 프레임을 한 번에 가져오기 (generate_motion용)
    std::vector<std::vector<double>> getNextSegment(int length, double ratio = 1.0) {
        std::vector<std::vector<double>> segment;
        for(int i=0; i<length; ++i) {
            Pose p = getNextPose(ratio);
            segment.push_back({p.r, p.p, p.y});
        }
        return segment;
    }

private:
    std::vector<Pose> frames;
    std::atomic<size_t> currentIndex{0}; // 쓰레드 간 공유되는 인덱스

    IdleMotionManager() = default;
    ~IdleMotionManager() = default;
    IdleMotionManager(const IdleMotionManager&) = delete;
    IdleMotionManager& operator=(const IdleMotionManager&) = delete;
};

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

#ifdef MOTOR_ENABLED
bool initialize_dynamixel() {
    // 1. 드라이버 생성
    dxl_driver = new DynamixelDriver(cfg_dxl.device_name, cfg_dxl.protocol_version, cfg_dxl.ids);


    // 2. 연결 (Baudrate 설정 포함)
    if (!dxl_driver->connect(cfg_dxl.baudrate)) {
        std::cerr << "Failed to connect to Dynamixel!" << std::endl;
        return false;
    }


    // 3. 기본 설정 (Torque Off 후 진행)
    dxl_driver->setTorque(false);


    if (!dxl_driver->setOperatingMode(cfg_dxl.operating_mode)) return false;
    if (!dxl_driver->setDriveMode(cfg_dxl.is_time_based)) return false;
    if (!dxl_driver->setReturnDelayTime(cfg_dxl.return_delay_time)) return false;


    // 4. PID 및 프로파일 설정
    if (!dxl_driver->setProfile(cfg_dxl.profile_velocity, cfg_dxl.profile_acceleration)) return false;
    if (!dxl_driver->setPositionPID(cfg_dxl.pos_p_gain, cfg_dxl.pos_i_gain, cfg_dxl.pos_d_gain)) return false;


    // 5. 토크 켜기
    if (!dxl_driver->setTorque(true)) {
        std::cerr << "Failed to enable torque!" << std::endl;
        return false;
    }

    printf("Motors initialized (Port Open, Torque On).\n");
    return true;
}

void move_to_initial_position_posctrl() {
    if (!dxl_driver) return;

    std::vector<int32_t> DXL_initial_position = { g_home.home_pitch, g_home.home_roll_r, g_home.home_roll_l, g_home.home_yaw, g_home.home_mouth };

    dxl_driver->writeGoalPosition(DXL_initial_position);
}


void move_to_initial_position_velctrl() {
    if (!dxl_driver) return;

    std::vector<int32_t> DXL_initial_position = { g_home.home_pitch, g_home.home_roll_r, g_home.home_roll_l, g_home.home_yaw, g_home.home_mouth };

    const int POSITION_TOLERANCE = 20; // 목표 위치 도달로 간주할 허용 오차
    const double P_GAIN = 0.2; // 비례 제어 상수 (이 값을 조절하여 감속 강도 변경)
    const int MAX_VELOCITY = 100; // 최대 속도 제한
    const int MIN_VELOCITY = 30;  // 최소 구동 속도

    std::vector<int32_t> goal_velocity(DXL_NUM, 0);

    printf("Moving to initial position...\n");

    while (true) {
        // 1. 현재 위치 읽기
        std::vector<MotorState> current_state;
        if (!dxl_driver->readAllState(current_state)) {
            std::cerr << "Failed to read motor states!" << std::endl;
            return;
        }

        bool all_motors_in_position = true;
        for (int i = 0; i < DXL_NUM; i++) {
            int position_diff = DXL_initial_position[i] - current_state[i].position;
            std::cout << "Motor " << cfg_dxl.ids[i] << " Diff: " << position_diff << std::endl;

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

                goal_velocity[i] = calculated_velocity;
            } else {
                goal_velocity[i] = 0; // 목표 도달 시 정지
            }
        }

        // 4. 계산된 목표 속도를 모터에 명령
        dxl_driver->writeGoalVelocity(goal_velocity);

        // 모든 모터가 목표 위치에 도달하면 루프 종료
        if (all_motors_in_position) {
            printf("Initial position reached.\n");
            break;
        }

        // 제어 주기 맞추기
        std::this_thread::sleep_for(std::chrono::milliseconds(CONTROL_MS));
    }

    // 최종적으로 모터 정지 명령
    for (int i = 0; i < DXL_NUM; i++) goal_velocity[i] = 0;

    dxl_driver->writeGoalVelocity(goal_velocity);
}
#endif // MOTOR_ENABLED

// 첫 번째 쓰레드: 오디오 스트림을 받아 분할합니다.
void stream_and_split(const SF_INFO& sfinfo, CustomSoundStream& soundStream) {
    std::vector<uint8_t>* buffer = &responses_stream_buffer;
    std::mutex* buffer_mutex = &responses_stream_buffer_mutex;
    std::condition_variable* buffer_cv = &responses_stream_buffer_cv;
    std::atomic<bool>* is_streaming_flag = &is_responses_streaming;

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
                return buffer->size() >= bytes_per_interval || !(*is_streaming_flag) || user_interruption_flag;
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

        // {
        //     auto now = std::chrono::high_resolution_clock::now();
        //     std::lock_guard<std::mutex> lock(cout_mutex);
        //     std::cout << "Stream and split cycle " << cycle_num << " at "
        //               << std::chrono::duration_cast<std::chrono::milliseconds>(now - start_time).count()
        //               << " ms" << std::endl;
        // }
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

        // {
        //     auto now = std::chrono::high_resolution_clock::now();
        //     std::lock_guard<std::mutex> lock(cout_mutex);
        //     std::cout << "Read and split cycle " << cycle_num << " at "
        //               << std::chrono::duration_cast<std::chrono::milliseconds>(now - start_time).count()
        //               << " ms" << std::endl;
        // }
    }

    // --- 4. 종료 처리 ---
    // 모든 처리가 끝났음을 후속 스레드에 알립니다.
    stop_flag = true;
    audio_queue_cv.notify_one(); // 대기 중인 generate_motion 스레드를 깨워 종료 조건을 확인시킵니다.
    if (vocal_sndfile) sf_close(vocal_sndfile);
}


void generate_motion(int channels, int samplerate) {

    std::vector<float> audio_buffer;

    // -------------------------
    // PART B mouth state
    // -------------------------
    PartBMouthState mouth_state;
    init_partb_config(mouth_state.cfg);

    // 실제 실행용 튜닝값 설정
    mouth_state.cfg.peak_trigger_min = 0.24f;
    mouth_state.cfg.prominence_th    = 0.13f;
    mouth_state.cfg.min_open         = 0.52f;
    mouth_state.cfg.future_steps     = 9;
    mouth_state.cfg.future_frames    = 9;   // peak center 9-point SG용 (360ms)

    init_partb_mouth_state(mouth_state);

    int frames_per_update = samplerate * 40 / 1000;

    std::vector<double> prevEndOneBefore = {0.0, 0.0, 0.0};
    std::vector<double> prevEnd          = {0.0, 0.0, 0.0};
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

        // ------------------------------------------------------------
        // 오디오 데이터 pop + 미래 peek 확보
        // ------------------------------------------------------------
        std::vector<float> next_peek;
        {
            std::unique_lock<std::mutex> lock(audio_queue_mutex);
            audio_queue_cv.wait(lock, [] { return !audio_queue.empty() || stop_flag || user_interruption_flag; });

            if ((stop_flag || user_interruption_flag) && audio_queue.empty()) {
                std::cout << "generate motion break ------------------------" << std::endl;
                break;
            }

            audio_buffer = std::move(audio_queue.front());
            audio_queue.pop();

            const int need_ms      = mouth_state.cfg.future_frames * mouth_state.cfg.hop_ms;
            const int need_frames  = samplerate * need_ms / 1000;
            const int need_samples = need_frames * channels;

            next_peek.clear();
            next_peek.reserve(std::max(0, need_samples));

            if (!audio_queue.empty() && need_samples > 0) {
                std::queue<std::vector<float>> tmp;
                tmp.swap(audio_queue);

                std::queue<std::vector<float>> restored;

                while (!tmp.empty()) {
                    auto& buf = tmp.front();

                    if ((int)next_peek.size() < need_samples) {
                        const int need = need_samples - (int)next_peek.size();
                        const int take = std::min<int>(need, (int)buf.size());
                        if (take > 0) {
                            next_peek.insert(next_peek.end(), buf.begin(), buf.begin() + take);
                        }
                    }

                    restored.push(std::move(buf));
                    tmp.pop();
                }

                audio_queue.swap(restored);
            }
        }

        // ============================================================
        // (A) PART B mouth trajectory 생성
        // ============================================================
        std::vector<float> out_a0_curr;

        build_partb_mouth_chunk(
            audio_buffer,
            next_peek,
            channels,
            samplerate,
            num_motion_updates,
            mouth_state,
            out_a0_curr
        );

        const int Ncurr = static_cast<int>(out_a0_curr.size());

        // ============================================================
        // (B) PART B a0 -> mouth tick
        // ============================================================
        std::vector<float> motion_results;
        motion_results.reserve(out_a0_curr.size());

        for (float a0_val : out_a0_curr) {
            float mouth_value = calculate_mouth(
                std::max(0.0f, a0_val),
                cfg_robot.max_mouth,
                cfg_robot.min_mouth
            );
            motion_results.push_back(mouth_value);
        }

        // ============================================================
        // (B-1) 전체 3점 스무딩
        // ============================================================
        if (motion_results.size() >= 3) {
            std::vector<float> motion_results_smooth = motion_results;

            for (size_t j = 1; j + 1 < motion_results.size(); ++j) {
                motion_results_smooth[j] =
                    (motion_results[j - 1] +
                    2.0f * motion_results[j] +
                    motion_results[j + 1]) / 4.0f;
            }

            motion_results.swap(motion_results_smooth);
        }

        // ============================================================
        // Head energy 계산
        // ============================================================
        energy.reserve(Ncurr);

        for (int i = 0; i < Ncurr; ++i) {
            int start_frame = i * frames_per_update;
            int end_frame   = start_frame + frames_per_update;

            int total_frames = static_cast<int>(audio_buffer.size()) / channels;
            if (end_frame > total_frames) end_frame = total_frames;
            if (start_frame >= total_frames) break;

            std::vector<float> current_audio(
                audio_buffer.begin() + start_frame * channels,
                audio_buffer.begin() + end_frame   * channels
            );

            std::vector<float> channel_divided =
                divide_channel(current_audio, channels, end_frame - start_frame);

            double rms_value = calculateRMS(channel_divided, 0, end_frame - start_frame);
            energy.push_back(static_cast<float>(rms_value));
        }

        // ============================================================
        // 첫 세그먼트 입 시작값 블렌딩
        // ============================================================
        if (!energy.empty()) {
            if (first_segment_flag == 1) {
                double start_mouth = 0.0;
                {
                    std::lock_guard<std::mutex> lock(prev_values_mutex);
                    prevSegment.clear();
                    for (const auto& val : prevValues) {
                        prevSegment.push_back({val[0], val[1], val[2]});
                    }
                    start_mouth = prevValues.back()[3];
                }

                int blend_frames = std::min<int>(5, motion_results.size());
                for (int k = 0; k < blend_frames; ++k) {
                    double t = static_cast<double>(k + 1) / static_cast<double>(blend_frames);
                    double alpha = t * t * (3.0 - 2.0 * t);

                    motion_results[k] = static_cast<float>(
                        start_mouth * (1.0 - alpha) + motion_results[k] * alpha
                    );
                }
            }

            // ============================================================
            // Head motion 생성
            // ============================================================
            if (cfg_robot.generate_head_motion) {
                avg_grad = getSegmentAverageGrad(energy, "one2one", "abs");
                segClass = assignClassWith1DMiddleBoundary(avg_grad, boundaries);

                std::string filePath;
                switch (segClass) {
                    case 0: filePath = "segment_0.npy"; break;
                    case 1: filePath = "segment_1.npy"; break;
                    case 2: filePath = "segment_2.npy"; break;
                    case 3: filePath = "segment_3.npy"; break;
                    default:
                        std::cerr << "Invalid segClass: " << segClass << std::endl;
                        break;
                }

                cnpy::NpyArray segment = cnpy::npy_load(SEGMENTS_DIR + "/" + filePath);

                for (int j = 0; j < 3; j++) {
                    prevEnd[j]          = prevSegment[prevSegment.size() - 1][j];
                    prevEndOneBefore[j] = prevSegment[prevSegment.size() - 2][j];
                }

                deliverSegment = getNextSegment_SegSeg(prevEndOneBefore, prevEnd, segment, true, true);
                deliverSegment = multExpToSegment(energy, deliverSegment, 0.01, 10);
                deliverSegment = connectTwoSegments(prevSegment, deliverSegment, 3, 3, 3);
                prevSegment = deliverSegment;
            } else {
                std::cout << "Idle motion 사용 중..." << std::endl;
                deliverSegment = IdleMotionManager::getInstance().getNextSegment(
                    energy.size(), cfg_robot.control_motor_rpy_ratio
                );

                if (first_segment_flag == 1) {
                    deliverSegment = connectTwoSegments(prevSegment, deliverSegment, 5, 3, 3);
                }
            }

            first_segment_flag = 0;
        }

        // ============================================================
        // queue push
        // ============================================================
        {
            std::lock_guard<std::mutex> lock(mouth_motion_queue_mutex);
            for (const auto& result : motion_results) {
                mouth_motion_queue.push(std::make_pair(cycle_num, result));
            }
            head_motion_queue.push(deliverSegment);
        }
        mouth_motion_queue_cv.notify_one();
    }
}


void control_motor(CustomSoundStream& soundStream, std::string mode_label) {
    #ifdef MOTOR_ENABLED
    std::vector<int32_t> past_position = dxl_driver->getLastGoalPosition();
    std::vector<int32_t> target_position(DXL_NUM);
    std::vector<int32_t> target_velocity(DXL_NUM);
    std::vector<MotorState> current_state(DXL_NUM);
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

        if (cycle_num == 0) {
            start_time = std::chrono::high_resolution_clock::now();
            soundStream.play(); // 첫 사이클에서 오디오 재생
            // Python에 playback_started 이벤트 전송
            send_to_python({{"type", "playback_started"}});
            {
                std::lock_guard<std::mutex> lock(cout_mutex);
                std::cout << "[시간 측정] start → 오디오 재생 시작: "
                        << std::chrono::duration_cast<std::chrono::milliseconds>(
                               std::chrono::high_resolution_clock::now() - start_time).count()
                        << "ms" << std::endl;
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


            
            float motor_value = motion_data.second;
            double roll  = 0;
            double pitch = 0;
            double yaw   = 0;
            double mouth = motor_value;

            #ifdef MOTOR_ENABLED
            target_position = RPY2DXL(roll, pitch, yaw, mouth, 0);

            if (first_move_flag == 1) {
                first_move_flag = 0;
            } else {
                for (int k = 0; k < DXL_NUM; k++) {
                    if (k == 4) continue; // mouth 축은 추가 스무딩 없음
                    target_position[k] = (past_position[k] + target_position[k]) / 2;
                }
            }

            // 상태 읽기
            dxl_driver->readAllState(current_state);

            // 모터 구동
            if (cfg_dxl.operating_mode == 1) {
                // 속도제어 모드
                for (int k = 0; k < DXL_NUM; k++) {
                    target_velocity[k] = calculateDXLGoalVelocity_timeBased_ds(current_state[k].position, target_position[k], current_state[k].velocity, cfg_dxl.profile_acceleration, CONTROL_MS);
                }
                dxl_driver->writeGoalVelocity(target_velocity);
            }
            else {
                // 위치제어 모드
                dxl_driver->writeGoalPosition(target_position);
            }

            // 과거 위치 업데이트
            past_position = target_position;
            updatePrevValues(roll, pitch, yaw, mouth);

            // 로깅
            double DXL_goal_rpy[4] = {roll, pitch, yaw, mouth};
            motion_logger.log(mode_label, DXL_goal_rpy, target_position, current_state);
            #else
            updatePrevValues(roll, pitch, yaw, mouth);
            #endif

            // 필요한 경우 대기 시간 추가
            // std::this_thread::sleep_for(std::chrono::milliseconds(39));
            std::this_thread::sleep_until(start_time + std::chrono::milliseconds(cycle_num * INTERVAL_MS + i * 40 + 40));
        }
    }
}

void wait_control_motor(){
    // 모터 초기 설정 코드
    if(wait_mode_flag == false) return;
    while(!mouth_motion_queue.empty()) mouth_motion_queue.pop();
    while(!head_motion_queue.empty()) head_motion_queue.pop();

    #ifdef MOTOR_ENABLED
    std::vector<int32_t> past_position = dxl_driver->getLastGoalPosition();
    std::vector<int32_t> target_position(DXL_NUM);
    std::vector<int32_t> target_velocity(DXL_NUM);
    std::vector<MotorState> current_state(DXL_NUM);

    std::cout << "대기 모드 (wait_control_motor) 시작: " << get_time_str() << std::endl;
    #else
    // --- 가짜 모터 초기화 ---
    std::cout << "[DUMMY MOTOR] 대기 모드 (wait_control_motor) 시작." << std::endl;
    #endif

    // std::this_thread::sleep_for(std::chrono::milliseconds(200));

    auto wait_start_time = std::chrono::high_resolution_clock::now();
    int step = 0;
    constexpr auto FRAME_INTERVAL = std::chrono::milliseconds(35);

    // -- 보간용 데이터 준비 --
    double ratio = cfg_robot.wait_mode_rpy_ratio;
    int SKIP_FRAMES = 20;

    // 초반 20프레임 가져오기 (Raw data: R, P, Y)
    auto rawSegment = IdleMotionManager::getInstance().getNextSegment(SKIP_FRAMES, ratio);
    
    // 보간을 위해 크기(4) 맞추기
    std::vector<std::vector<double>> targetTraj;
    for(const auto& pose : rawSegment) {
        // prevValues 구조(R, P, Y, M)에 맞춤
        targetTraj.push_back({pose[0], pose[1], pose[2], 0.0});
    }

    // 시작 포즈 가져오기
    std::vector<double> startPose;
    {
        std::lock_guard<std::mutex> lock(prev_values_mutex);
        startPose = prevValues.back();
    }

    // 보간 적용
    targetTraj = applyOffsetDecay(startPose, targetTraj, SKIP_FRAMES);

    while(wait_mode_flag == true){
        #ifdef MOTOR_ENABLED

        // 모션 재생
        double roll_final, pitch_final, yaw_final, mouth_final;

        if (step < SKIP_FRAMES) {
            // 보간된 구간 재생 (이미 ratio 적용됨)
            roll_final = targetTraj[step][0];
            pitch_final = targetTraj[step][1];
            yaw_final = targetTraj[step][2];
            mouth_final = targetTraj[step][3];
        } 
        else {
            // 보간 이후 IdleMotionManager에서 계속 가져오기 (ratio 적용)
            auto pose = IdleMotionManager::getInstance().getNextPose(ratio);
            roll_final = pose.r;
            pitch_final = pose.p;
            yaw_final = pose.y;
            mouth_final = 0.0;
        }

        target_position = RPY2DXL(roll_final, pitch_final, yaw_final, mouth_final, 0);

        // 상태 읽기
        dxl_driver->readAllState(current_state);

        // 모터 구동
        if (cfg_dxl.operating_mode == 1) {
            // 속도제어 모드
            for (int i = 0; i < DXL_NUM; i++) {
                target_velocity[i] = calculateDXLGoalVelocity_timeBased_ds(current_state[i].position, target_position[i], current_state[i].velocity, cfg_dxl.profile_acceleration, 35);
            }
            
            dxl_driver->writeGoalVelocity(target_velocity);
        }
        else {
            // 위치제어 모드
            dxl_driver->writeGoalPosition(target_position);
        }

        // 과거 위치 업데이트
        past_position = target_position;
        updatePrevValues(roll_final, pitch_final, yaw_final, mouth_final);

        // 로깅
        double DXL_goal_rpy[4] = {roll_final, pitch_final, yaw_final, mouth_final};
        motion_logger.log("WAIT", DXL_goal_rpy, target_position, current_state);
        
        step ++;
        std::this_thread::sleep_until(wait_start_time + FRAME_INTERVAL * step);
        // std::this_thread::sleep_for(std::chrono::milliseconds(34));
        
        #else
        // --- 가짜 모터 대기 동작 ---
        if(wait_mode_flag == false) break;
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        #endif
    }
    std::cout << "wait mode finish " << std::endl;
}

// 모션 csv 파일 읽어서 재생하는 함수 (테스트용)
void csv_control_motor(std::string audioName) {
    while(!mouth_motion_queue.empty()) mouth_motion_queue.pop();
    while(!head_motion_queue.empty()) head_motion_queue.pop();

    #ifdef MOTOR_ENABLED
    std::vector<int32_t> past_position = dxl_driver->getLastGoalPosition();
    std::vector<int32_t> target_position(DXL_NUM);
    std::vector<int32_t> target_velocity(DXL_NUM);
    std::vector<MotorState> current_state(DXL_NUM);

    std::cout << "모션 CSV 재생 모드 (csv_control_motor) 시작." << std::endl;
    #else
    std::cout << "[DUMMY MOTOR] 모션 CSV 재생 모드 (csv_control_motor) 시작." << std::endl;
    #endif

    sf::Music music;

    std::string headMotionFilePath = "assets/headMotion/" + audioName + ".csv";
    std::string mouthMotionFilePath = "assets/mouthMotion/" + audioName + "-delta-big.csv";
    std::string audioFilePath = "assets/audio/music/" + audioName + ".wav";

    if (!music.openFromFile(audioFilePath)) {
        std::cerr << "Error: Could not load audio file: " << audioFilePath << std::endl;
        return;
    }

    auto csv_start_time = std::chrono::high_resolution_clock::now();
    int step = 0;
    constexpr auto FRAME_INTERVAL = std::chrono::milliseconds(40);

    while(true){
        #ifdef MOTOR_ENABLED
        std::ifstream headGesture(headMotionFilePath);
        if (!headGesture) {
            std::cerr << "HeadGesture File not found." << std::endl;
            return;
        }
        std::ifstream MouthGesture(mouthMotionFilePath);
        if (!MouthGesture) {
            std::cerr << "MouthGesture File not found." << std::endl;
            return;
        }

        // 초기 프레임 궤적 보간
        int SKIP_FRAMES = 20;
        std::vector<std::vector<double>> targetTraj;

        for (int i = 0; i < SKIP_FRAMES; i++) {
            if (!headGesture.good() || !MouthGesture.good()) break;
            auto headRow = csv_read_row(headGesture, ',');
            auto mouthRow = csv_read_row(MouthGesture, ',');
            float roll_s = std::stof(headRow[0]);
            float pitch_s = std::stof(headRow[1]);
            float yaw_s = std::stof(headRow[2]);
            float mouth_s = std::stof(mouthRow[0]);

            float ratiooo = std::stof(mouthRow[1]) * 1.4;

            targetTraj.push_back({roll_s * ratiooo, pitch_s * ratiooo, yaw_s * ratiooo, mouth_s});
        }

        std::vector<double> startPose;
        {
            std::lock_guard<std::mutex> lock(prev_values_mutex);
            startPose = prevValues.back();
        }

        std::cout << "Original trajectory:" << std::endl;
        for (const auto& pose : targetTraj) {
            std::cout << pose[0] << ", ";
        }
        std::cout << std::endl;

		targetTraj = applyOffsetDecay(startPose, targetTraj, SKIP_FRAMES);
        
        std::cout << "Interpolated trajectory:" << std::endl;
        for (const auto& pose : targetTraj) {
            std::cout << pose[0] << ", ";
        }
        std::cout << std::endl;

        // 모션 재생
        while(headGesture.good() && MouthGesture.good()){
            if (user_interruption_flag) {
                std::cout << "Interruption detected in csv_control_motor." << std::endl;
                music.stop();
                return;
            }

            if (music.getStatus() != sf::Music::Playing) {
                music.play();
            }

            double roll_final, pitch_final, yaw_final, mouth_final;

            if (step < SKIP_FRAMES) {
                roll_final = targetTraj[step][0];
                pitch_final = targetTraj[step][1];
                yaw_final = targetTraj[step][2];
                mouth_final = targetTraj[step][3];
            }
            else {
                auto headRow = csv_read_row(headGesture, ',');
                auto mouthRow = csv_read_row(MouthGesture, ',');
                
                float roll_s = std::stof(headRow[0]);
                float pitch_s = std::stof(headRow[1]);
                float yaw_s = std::stof(headRow[2]);
                float mouth_s = std::stof(mouthRow[0]);

                float ratiooo = std::stof(mouthRow[1]) * 1.4;

                roll_final = roll_s * ratiooo;
                pitch_final = pitch_s * ratiooo;
                yaw_final = yaw_s * ratiooo;
                mouth_final = mouth_s;
            }

            target_position = RPY2DXL(roll_final , pitch_final, yaw_final, mouth_final, 0);
            
            // 상태 읽기
            dxl_driver->readAllState(current_state);

            // 모터 제어
            if (cfg_dxl.operating_mode == 1) {
                // 속도제어 모드
                for (int i = 0; i < DXL_NUM; i++) {
                    target_velocity[i] = calculateDXLGoalVelocity_timeBased_ds(current_state[i].position, target_position[i], current_state[i].velocity, cfg_dxl.profile_acceleration, 35);
                }
                
                dxl_driver->writeGoalVelocity(target_velocity);
            }
            else {
                // 위치제어 모드
                dxl_driver->writeGoalPosition(target_position);
            }

            // 과거 위치 업데이트
            past_position = target_position;
            updatePrevValues(roll_final , pitch_final, yaw_final, mouth_final);

            // 로깅
            double DXL_goal_rpy[4] = {roll_final, pitch_final, yaw_final, mouth_final};
            motion_logger.log("PLAY_AUDIO_CSV", DXL_goal_rpy, target_position, current_state);
            
            // 동작과 소리 싱크 확인
            // sf::Int32 music_ms = music.getPlayingOffset().asMilliseconds();
            // std::cout << "Step: " << step << ", Motor ms: " << timestamp.count() << ", Music ms: " << music_ms << ", Diff: " << (timestamp.count() - music_ms) << "ms" << std::endl;
            
            // 제어 주기 맞추기
            std::this_thread::sleep_until(csv_start_time + FRAME_INTERVAL * step);
            step ++;
        }
        #else
        // --- 모터 비활성화됨 ---
        std::cout << "[DUMMY MOTOR] 모션 CSV 재생 모드 (csv_control_motor)." << std::endl;
        #endif
        music.stop();
        return;
    }
}


#ifdef MOTOR_ENABLED
static constexpr int MPU6050_ADDR = 0x68;

// MPU6050 초기화
void mpu6050_init(int fd) {
    wiringPiI2CWriteReg8(fd, 0x6B, 0);
}

// 16비트 데이터 읽기
int read_raw_data(int fd, int addr) {
    int high = wiringPiI2CReadReg8(fd, addr);
    int low = wiringPiI2CReadReg8(fd, addr + 1);
    int value = (high << 8) | low;

    if (value > 32768)
        value -= 65536;
    return value;
}

void gyro_test() {

    // 6) MPU6050 초기화
    if (wiringPiSetup() == -1) {
        std::cerr << "WiringPi 초기화 실패!" << std::endl;
        return;
    }
    int fd = wiringPiI2CSetup(MPU6050_ADDR);
    if (fd == -1) {
        std::cerr << "MPU6050 I2C 연결 실패!" << std::endl;
        return;
    }
    mpu6050_init(fd);
    std::cout << "MPU6050 데이터 수집 시작..." << std::endl;

    std::vector<int> DXL_goal_position;
    int Roll_L_adjust_flag = 0;
    int Roll_R_adjust_flag = 0;
    int Pitch_adjust_flag = 0;
    int mouth_adjust_flag = 0;

    const float current_threshold_mA = -20;   // 목표 전류 임계값 (mA)
    const int adjustment_increment = 3;       // 모터 위치 조정 증분 (펄스)
    bool tension_satisfied = false;
    const int sample_count = 3;

    std::cout << "Roll 조정" << std::endl;

    while (true) {
        int sum_accel_x = 0, sum_accel_y = 0, sum_accel_z = 0;
        for (int i = 0; i < sample_count; i++) {
            sum_accel_x += read_raw_data(fd, 0x3B);
            sum_accel_y += read_raw_data(fd, 0x3D);
            sum_accel_z += read_raw_data(fd, 0x3F);
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
    }
}

#include <fstream>
#include <filesystem>


void initialize_robot_posture() {
    // MPU6050 초기화
    if (wiringPiSetup() == -1) {
        std::cerr << "WiringPi 초기화 실패!" << std::endl;
        return;
    }
    int fd = wiringPiI2CSetup(MPU6050_ADDR);
    if (fd == -1) {
        std::cerr << "MPU6050 I2C 연결 실패!" << std::endl;
        return;
    }
    mpu6050_init(fd);
    std::cout << "MPU6050 데이터 수집 시작..." << std::endl;

    std::vector<int32_t> target_position = {g_home.home_pitch, g_home.home_roll_r, g_home.home_roll_l, g_home.home_yaw, g_home.home_mouth};
    bool Roll_L_adjust_flag = 0;
    bool Roll_R_adjust_flag = 0;
    bool Pitch_adjust_flag = 0;
    bool mouth_adjust_flag = 0;

    const float current_threshold_mA = -20;   // 목표 전류 임계값 (mA)
    const int adjustment_increment = 3;       // 모터 위치 조정 증분 (펄스)
    bool tension_satisfied = false;
    const int sample_count = 3;

    std::cout << "Roll 조정" << std::endl;

    int sum_accel_x = 0, sum_accel_y = 0, sum_accel_z = 0;
    for (int i = 0; i < sample_count; i++) {
        sum_accel_x += read_raw_data(fd, 0x3B);
        sum_accel_y += read_raw_data(fd, 0x3D);
        sum_accel_z += read_raw_data(fd, 0x3F);
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
    if (Ax > 0){
        // Roll_L 조정
        while(true){
            target_position[2] -= adjustment_increment;
            dxl_driver->writeGoalPosition(target_position);

            sum_accel_x = 0, sum_accel_y = 0, sum_accel_z = 0;
            for (int i = 0; i < sample_count; i++) {
                sum_accel_x += read_raw_data(fd, 0x3B);
                sum_accel_y += read_raw_data(fd, 0x3D);
                sum_accel_z += read_raw_data(fd, 0x3F);
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
            target_position[1] -= adjustment_increment;
            dxl_driver->writeGoalPosition(target_position);

            sum_accel_x = 0, sum_accel_y = 0, sum_accel_z = 0;
            for (int i = 0; i < sample_count; i++) {
                sum_accel_x += read_raw_data(fd, 0x3B);
                sum_accel_y += read_raw_data(fd, 0x3D);
                sum_accel_z += read_raw_data(fd, 0x3F);
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
            target_position[1] -= adjustment_increment;
            dxl_driver->writeGoalPosition(target_position);

            sum_accel_x = 0, sum_accel_y = 0, sum_accel_z = 0;
            for (int i = 0; i < sample_count; i++) {
                sum_accel_x += read_raw_data(fd, 0x3B);
                sum_accel_y += read_raw_data(fd, 0x3D);
                sum_accel_z += read_raw_data(fd, 0x3F);
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
            target_position[2] -= adjustment_increment;
            dxl_driver->writeGoalPosition(target_position);

            sum_accel_x = 0, sum_accel_y = 0, sum_accel_z = 0;
            for (int i = 0; i < sample_count; i++) {
                sum_accel_x += read_raw_data(fd, 0x3B);
                sum_accel_y += read_raw_data(fd, 0x3D);
                sum_accel_z += read_raw_data(fd, 0x3F);
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
        sum_accel_x += read_raw_data(fd, 0x3B);
        sum_accel_y += read_raw_data(fd, 0x3D);
        sum_accel_z += read_raw_data(fd, 0x3F);
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
            target_position[0] -= adjustment_increment;
            dxl_driver->writeGoalPosition(target_position);

            sum_accel_x = 0, sum_accel_y = 0, sum_accel_z = 0;
            for (int i = 0; i < sample_count; i++) {
                sum_accel_x += read_raw_data(fd, 0x3B);
                sum_accel_y += read_raw_data(fd, 0x3D);
                sum_accel_z += read_raw_data(fd, 0x3F);
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
            target_position[0] -= adjustment_increment;
            dxl_driver->writeGoalPosition(target_position);

            sum_accel_x = 0, sum_accel_y = 0, sum_accel_z = 0;
            for (int i = 0; i < sample_count; i++) {
                sum_accel_x += read_raw_data(fd, 0x3B);
                sum_accel_y += read_raw_data(fd, 0x3D);
                sum_accel_z += read_raw_data(fd, 0x3F);
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
            target_position[1] -= adjustment_increment;
            target_position[2] -= adjustment_increment;
            dxl_driver->writeGoalPosition(target_position);

            sum_accel_x = 0, sum_accel_y = 0, sum_accel_z = 0;
            for (int i = 0; i < sample_count; i++) {
                sum_accel_x += read_raw_data(fd, 0x3B);
                sum_accel_y += read_raw_data(fd, 0x3D);
                sum_accel_z += read_raw_data(fd, 0x3F);
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
    
    // =============================
    // Mouth 조정 (ΔI_raw(LSB) 기반 + MAD 자동 임계값 학습)
    // - 목적: 초기 캘리브레이션 단계에서 "전류 급변" 감지 시 즉시 멈춤(Backoff)
    // =============================

    // DataLogger MouthLogger

    // std::string log_dir = create_log_directory("logs/calibration/");
    // auto log_start_time = std::chrono::high_resolution_clock::now();
    // MouthLogger.start(log_start_time, log_dir);

    // =============================
    // Mouth 조정 (ΔI_raw(LSB) 기반 + MAD 자동 임계값 학습)
    // - 목적: 초기 캘리브레이션 단계에서 "전류 급변" 감지 시 즉시 멈춤(Backoff)
    // - present position 읽기 기능 없이(goal 기반) 동작
    // =============================

    std::cout << "Mouth 조정 (delta-current LSB + MAD auto threshold)" << std::endl;

    std::filesystem::create_directories("data");
    std::ofstream logf("data/log_only_mouth.csv", std::ios::out | std::ios::trunc);
    if (!logf.is_open()) {
        std::cerr << "CSV 열기 실패: data/log_only_mouth.csv\n";
        return;
    }
    logf.setf(std::ios::unitbuf);
    logf << "t_ms,mouth_goal,raw_current_LSB,current_mA,abs_delta_raw_LSB,abs_delta_mA,thr_raw_LSB\n";

    // ---- 설정값 ----
    const float mA_per_LSB   = 2.69f;
    const int   N_CUR        = 3;
    const int   CUR_DELAY_MS = 10;
    const int   SETTLE_MS    = 30;
    const int   MAX_STEPS    = 600;

    const int   MOUTH_STEP_TICK    = 3;
    const int   MOUTH_BACKOFF_TICK = 15;

    // 자동학습 파라미터
    const int   LEARN_STEPS      = 25;
    const float THR_MAD_K        = 8.0f;   // 6~10 권장
    const int   THR_MIN_RAW_LSB  = 2;
    const int   THR_MAX_RAW_LSB  = 20;

    // 연속 조건
    const int   HIT_COUNT = 1;

    // ---- 전류 raw(LSB) 읽기 (평균) ----
    std::vector<int16_t> current(5, 0);
    auto read_mouth_current_raw = [&]() -> int {
        long sum = 0;
        int got = 0;
        for (int k = 0; k < N_CUR; k++) {
            if (dxl_driver->readPresentCurrent(current)) {
                sum += current[4]; // mouth index 4
                got++;
            }
            delay(CUR_DELAY_MS);
        }
        if (got == 0) return 0;
        return (int)std::lround((double)sum / (double)got);
    };

    // ---- median / MAD 유틸 ----
    auto median_int = [](std::vector<int> v) -> int {
        if (v.empty()) return 0;
        size_t mid = v.size() / 2;
        std::nth_element(v.begin(), v.begin() + mid, v.end());
        int m = v[mid];
        if (v.size() % 2 == 0) {
            std::nth_element(v.begin(), v.begin() + mid - 1, v.end());
            m = (m + v[mid - 1]) / 2;
        }
        return m;
    };

    auto mad_int = [&](const std::vector<int>& v, int med) -> int {
        std::vector<int> dev;
        dev.reserve(v.size());
        for (int x : v) dev.push_back(std::abs(x - med));
        return median_int(std::move(dev));
    };

    // ---- 초기값 ----
    int prev_raw = read_mouth_current_raw();
    int thr_raw  = THR_MIN_RAW_LSB;
    int hit      = 0;

    // ---- 1) 임계값 자동 학습 ----
    std::vector<int> deltas;
    deltas.reserve(LEARN_STEPS);

    for (int i = 0; i < LEARN_STEPS; i++) {
        target_position[4] -= MOUTH_STEP_TICK;
        dxl_driver->writeGoalPosition(target_position);
        delay(SETTLE_MS);

        int cur_raw = read_mouth_current_raw();
        int d_raw   = std::abs(cur_raw - prev_raw);
        deltas.push_back(d_raw);

        float cur_mA = cur_raw * mA_per_LSB;
        float d_mA   = d_raw   * mA_per_LSB;

        // 학습 중 thr 미확정이므로 -1 기록
        logf << millis() << "," << target_position[4] << ","
            << cur_raw << "," << cur_mA << ","
            << d_raw << "," << d_mA << ","
            << -1 << "\n";

        prev_raw = cur_raw;
    }

    // MAD 기반 임계값
    int med = median_int(deltas);
    int mad = mad_int(deltas, med);

    int auto_thr = (int)std::ceil((double)med + (double)THR_MAD_K * (double)mad);
    thr_raw = std::max(auto_thr, THR_MIN_RAW_LSB);
    thr_raw = std::min(thr_raw, THR_MAX_RAW_LSB);

    std::cout << "[Mouth] learned thr_raw=" << thr_raw
            << " (median=" << med << ", mad=" << mad << ")\n";

    // ---- 2) 본 탐색 ----
    mouth_adjust_flag = false;
    hit = 0;

    for (int step = 0; step < MAX_STEPS && !mouth_adjust_flag; step++) {
        target_position[4] -= MOUTH_STEP_TICK;
        dxl_driver->writeGoalPosition(target_position);
        delay(SETTLE_MS);

        int cur_raw = read_mouth_current_raw();
        int d_raw   = std::abs(cur_raw - prev_raw);

        float cur_mA = cur_raw * mA_per_LSB;
        float d_mA   = d_raw   * mA_per_LSB;

        logf << millis() << "," << target_position[4] << ","
            << cur_raw << "," << cur_mA << ","
            << d_raw << "," << d_mA << ","
            << thr_raw << "\n";

        if (d_raw >= thr_raw) {
            hit++;
        } else {
            hit = 0;
        }

        if (hit >= HIT_COUNT) {
            // goal 기준 backoff (present 없으니 기존 방식)
            target_position[4] += MOUTH_BACKOFF_TICK;
            dxl_driver->writeGoalPosition(target_position);
            delay(150);

            mouth_adjust_flag = true;
            break;
        }

        prev_raw = cur_raw;
    }

    logf.flush();
    logf.close();

    // 결과 저장
    g_home.home_pitch  = target_position[0];
    g_home.home_roll_r = target_position[1];
    g_home.home_roll_l = target_position[2];
    g_home.home_yaw    = target_position[3];
    g_home.home_mouth  = target_position[4];

    finish_adjust_ready = true;

}

void cleanup_dynamixel() {
    std::cout << "토크를 끄고 포트를 닫습니다..." << std::endl;
    if (dxl_driver) {
        delete dxl_driver;
        dxl_driver = nullptr;
    }
}
#endif // MOTOR_ENABLED

void signal_handler(int signum) {
    std::cout << "종료 신호 (" << signum << ") 수신. 프로그램을 정리합니다." << std::endl;
    
    stop_flag = true;
    wait_mode_flag = false;
    user_interruption_flag = true;

    server_message_queue_cv.notify_all();
    audio_queue_cv.notify_all();
    mouth_motion_queue_cv.notify_all();
    responses_stream_buffer_cv.notify_all();

    if (g_ws_server) g_ws_server->stop();

    #ifdef MOTOR_ENABLED
    if (tuning_logger) tuning_logger->stop();
    motion_logger.stop();
    cleanup_dynamixel();
    #endif

    std::_Exit(signum);
}

// 큐 초기화용 함수
auto clear_queues() {
    {
        std::lock_guard<std::mutex> lock(audio_queue_mutex);
        std::queue<std::vector<float>> empty;
        std::swap(audio_queue, empty);
    }
    {
        std::lock_guard<std::mutex> lock(mouth_motion_queue_mutex);
        std::queue<std::pair<int, float>> empty_mouth;
        std::swap(mouth_motion_queue, empty_mouth);
        std::queue<std::vector<std::vector<double>>> empty_head;
        std::swap(head_motion_queue, empty_head);
    }
}

void robot_main_loop(std::future<void> server_ready_future) {
    std::cout << "Python 클라이언트 연결 대기 중..." << std::endl;
    server_ready_future.get(); // 클라이언트가 연결될 때까지 대기
    std::cout << "Python 클라이언트 연결 완료!" << std::endl;

    #ifdef MOTOR_ENABLED
    std::string log_dir = create_log_directory();
    auto log_start_time = std::chrono::high_resolution_clock::now();
    motion_logger.start(log_start_time, log_dir);
    if (tuning_logger) tuning_logger->start(log_start_time, log_dir);
    #endif

	std::thread wait_mode_thread;

    std::pair<std::string,std::string> play_music;
    while (true) {
        // --- 루프 시작 시 상태 초기화 ---
        stop_flag = false;
        is_responses_streaming = false;
        {
            std::lock_guard<std::mutex> lock(responses_stream_buffer_mutex);
            responses_stream_buffer.clear();
        }
        
        SF_INFO sfinfo;
        SNDFILE* sndfile = nullptr;
        bool is_file_based = false;
        bool is_csv_based = false;
        std::string csv_audio_name = "";
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

            if (type == "play_file") {
                current_mode_label = "PLAY_AUDIO";
                std::string file_path = response.value("file_path", "");
                sndfile = sf_open(file_path.c_str(), SFM_READ, &sfinfo);
                if (sndfile) is_file_based = true;
            }
            else if (type == "play_music") {
                current_mode_label = "PLAY_MUSIC";
                music_flag = 1;
                std::string file_to_play = response.value("file_to_play", "");
                play_music = {response.value("title", ""), response.value("artist", "")};
                sndfile = sf_open(file_to_play.c_str(), SFM_READ, &sfinfo);
                if (sndfile) is_file_based = true;
            }
            else if (type == "stream_start") {
                is_file_based = false;
                current_mode_label = "RESPONSE";
                sfinfo.channels = AUDIO_CHANNELS;
                sfinfo.samplerate = AUDIO_SAMPLE_RATE;
            }
            else if (type == "play_audio_csv") {
                is_csv_based = true;
                csv_audio_name = response.value("audio_name", "");
            }
            else {
                std::cerr << "Error: Unknown command type received: " << type << std::endl;
            }
        }

        std::cout << "[MainLoop] is_file_based=" << is_file_based << " is_csv=" << is_csv_based << " sndfile=" << (sndfile ? "OK" : "NULL") << std::endl;
        CustomSoundStream soundStream(sfinfo.channels, sfinfo.samplerate);
        std::cout << "[MainLoop] SoundStream 생성 완료" << std::endl;

        // --- 2. 스레드 시작 ---
        is_speaking = true;
        clear_queues();

        if (is_csv_based) {
            wait_mode_flag = false;
			if (wait_mode_thread.joinable()) {
				wait_mode_thread.join();
			}
            csv_control_motor(csv_audio_name);
        }
        else if (is_file_based) {
			wait_mode_flag = false;
			std::cout << "[MainLoop] wait_mode join 시작" << std::endl;
			if (wait_mode_thread.joinable()) {
				wait_mode_thread.join();
			}
			std::cout << "[MainLoop] wait_mode join 완료, 재생 시작" << std::endl;
            start_time = std::chrono::high_resolution_clock::now();
            std::thread t1(read_and_split, sndfile, sfinfo, std::ref(soundStream));
            std::thread t2(generate_motion, sfinfo.channels, sfinfo.samplerate);
            std::thread t3(control_motor, std::ref(soundStream), current_mode_label);
            t1.join();
            t2.join();
            t3.join();
        }
        else { // responses 스트리밍
            const size_t bytes_per_interval = sfinfo.samplerate * sfinfo.channels * sizeof(sf::Int16) * INTERVAL_MS / 1000;

            // Responses 처리
            if (!user_interruption_flag) {
                // audio 데이터가 들어올 때까지 대기
                {
                    std::unique_lock<std::mutex> lock(responses_stream_buffer_mutex);
                    responses_stream_buffer_cv.wait(lock, [&]{ return responses_stream_buffer.size() >= bytes_per_interval || !is_responses_streaming || user_interruption_flag; });
                }

                if (!responses_stream_buffer.empty() && !user_interruption_flag) {
                    wait_mode_flag = false;
					if (wait_mode_thread.joinable()) {
						wait_mode_thread.join();
					}

                    stop_flag = false;
                    start_time = std::chrono::high_resolution_clock::now();
                    std::thread t1_responses(stream_and_split, std::ref(sfinfo), std::ref(soundStream));
                    std::thread t2_responses(generate_motion, sfinfo.channels, sfinfo.samplerate);
                    std::thread t3_responses(control_motor, std::ref(soundStream), "RESPONSES");

                    t1_responses.join();
                    t2_responses.join();
                    t3_responses.join();
                }
            }
        }

        // playback_complete는 정상 종료/인터럽션 구분 없이 항상 1회 전송
        if (user_interruption_flag) {
            std::cout << "Interruption handling: Cleaning up resources." << std::endl;
        }
        std::cout << "[MainLoop] playback_complete 전송" << std::endl;
        send_to_python({{"type", "playback_complete"}});

        // 리소스 정리
        soundStream.stop();
        soundStream.clearBuffer();
        clear_queues();
        
        if (sndfile) sf_close(sndfile);
        playing_music_flag = false;

        is_speaking = false;
    }
}

int main(int argc, char* argv[]) {
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);

    LoadConfig("cpp/config.toml");

    g_home.home_pitch  = cfg_robot.default_pitch;
    g_home.home_roll_r = cfg_robot.default_roll_r;
    g_home.home_roll_l = cfg_robot.default_roll_l;
    g_home.home_yaw    = cfg_robot.default_yaw;
    g_home.home_mouth  = cfg_robot.default_mouth;

    // Idle Motion 파일 로드
    if (!IdleMotionManager::getInstance().loadCSV(IDLE_MOTION_FILE)) {
        std::cerr << "Idle motions 로드 실패!" << std::endl;
        return -1;
    }

    #ifdef MOTOR_ENABLED
    if (!initialize_dynamixel()) {
        std::cerr << "모터 초기화 실패!" << std::endl;
        return -1;
    }

    // 초기 자세로 이동
    if (cfg_dxl.operating_mode == 1)
        move_to_initial_position_velctrl();
    else {
        dxl_driver->setProfile(cfg_dxl.profile_velocity_homing, cfg_dxl.profile_acceleration);
        move_to_initial_position_posctrl();
        dxl_driver->setProfile(cfg_dxl.profile_velocity, cfg_dxl.profile_acceleration);
    }

    std::this_thread::sleep_for(std::chrono::seconds(1));

    // 자이로센서를 이용한 로봇 초기자세 설정
    // dxl_driver->setProfile(cfg_dxl.profile_velocity_homing, cfg_dxl.profile_acceleration);
    // initialize_robot_posture();
    // dxl_driver->setProfile(cfg_dxl.profile_velocity, cfg_dxl.profile_acceleration);

    // gyro_test();

    tuning_logger = new HighFreqLogger(dxl_driver);
    #endif

    // ===== CSV 재생 모드: ./build/Ray --csv <이름> =====
    if (argc >= 3 && std::string(argv[1]) == "--csv") {
        std::string audioName = argv[2];
        std::cout << "[MAIN] CSV 재생 모드: " << audioName << std::endl;

        std::string log_dir = create_log_directory();
        auto log_start_time = std::chrono::high_resolution_clock::now();
        motion_logger.start(log_start_time, log_dir);
        #ifdef MOTOR_ENABLED
        if (tuning_logger) tuning_logger->start(log_start_time, log_dir);
        #endif

        csv_control_motor(audioName);

        #ifdef MOTOR_ENABLED
        if (tuning_logger) tuning_logger->stop();
        #endif
        motion_logger.stop();
        cleanup_dynamixel();
        return 0;
    }

    // ===== 오디오 파일 재생 모드: ./build/Ray --play <파일경로> =====
    if (argc >= 3 && std::string(argv[1]) == "--play") {
        std::string filePath = argv[2];
        std::cout << "[MAIN] 오디오 파일 재생 모드: " << filePath << std::endl;

        SF_INFO sfinfo = {};
        SNDFILE* sndfile = sf_open(filePath.c_str(), SFM_READ, &sfinfo);
        if (!sndfile) {
            std::cerr << "오디오 파일 열기 실패: " << filePath << std::endl;
            cleanup_dynamixel();
            return -1;
        }

        std::string log_dir = create_log_directory();
        auto log_start_time = std::chrono::high_resolution_clock::now();
        motion_logger.start(log_start_time, log_dir);
        #ifdef MOTOR_ENABLED
        if (tuning_logger) tuning_logger->start(log_start_time, log_dir);
        #endif

        CustomSoundStream soundStream(sfinfo.channels, sfinfo.samplerate);
        start_time = std::chrono::high_resolution_clock::now();

        std::thread t1(read_and_split, sndfile, sfinfo, std::ref(soundStream));
        std::thread t2(generate_motion, sfinfo.channels, sfinfo.samplerate);
        std::thread t3(control_motor, std::ref(soundStream), "PLAY_FILE");

        t1.join();
        t2.join();
        t3.join();

        soundStream.stop();
        soundStream.clearBuffer();
        sf_close(sndfile);

        #ifdef MOTOR_ENABLED
        if (tuning_logger) tuning_logger->stop();
        #endif
        motion_logger.stop();
        cleanup_dynamixel();
        return 0;
    }

    // ===== 기본 모드: 웹소켓 서버 =====

    // 웹소켓 서버 준비 (Python 클라이언트가 접속)
    std::future<void> server_ready_future = server_ready_promise.get_future();
    ix::initNetSystem();

    const int ws_port = cfg_ws.port;
    ix::WebSocketServer ws_server(ws_port, "0.0.0.0");
    g_ws_server = &ws_server;

    ws_server.setOnClientMessageCallback(
        [](std::shared_ptr<ix::ConnectionState> connectionState,
           ix::WebSocket& ws,
           const ix::WebSocketMessagePtr& msg) {

        if (msg->type == ix::WebSocketMessageType::Open) {
            std::cout << "[WebSocket] Python 클라이언트 연결됨: " << connectionState->getRemoteIp() << std::endl;
            {
                std::lock_guard<std::mutex> lock(g_client_ws_mutex);
                g_client_ws = &ws;
            }
            // promise는 한 번만 set 가능 — 재접속 시 중복 set 방지
            bool expected = false;
            if (server_ready_fired.compare_exchange_strong(expected, true)) {
                server_ready_promise.set_value();
            }
        }
        else if (msg->type == ix::WebSocketMessageType::Close) {
            std::cout << "[WebSocket] Python 클라이언트 연결 해제" << std::endl;
            {
                std::lock_guard<std::mutex> lock(g_client_ws_mutex);
                g_client_ws = nullptr;
            }
        }
        else if (msg->type == ix::WebSocketMessageType::Message) {
            try {
                json response = json::parse(msg->str);
                std::string type = response.value("type", "");

                if (type == "audio") {
                    if (user_interruption_flag) return;
                    std::string b64_data = response.value("data", "");
                    std::string decoded_data;
                    macaron::Base64::Decode(b64_data, decoded_data);
                    std::lock_guard<std::mutex> lock(responses_stream_buffer_mutex);
                    responses_stream_buffer.insert(responses_stream_buffer.end(), decoded_data.begin(), decoded_data.end());
                    responses_stream_buffer_cv.notify_one();
                }
                else if (type == "audio_end") {
                    is_responses_streaming = false;
                    responses_stream_buffer_cv.notify_one();
                }
                else if (type == "stop") {
                    if (is_speaking) {
                        std::cout << "[WebSocket] Stop (interrupt) received." << std::endl;
                        user_interruption_flag = true;
                        responses_stream_buffer_cv.notify_all();
                        audio_queue_cv.notify_all();
                        mouth_motion_queue_cv.notify_all();
                    }
                }
                else {
                    // stream_start, play_file, play_music, play_audio_csv → 메인 루프가 처리
                    if (type == "stream_start" || type == "play_file" || type == "play_music" || type == "play_audio_csv") {
                        user_interruption_flag = false;
                        if (type == "stream_start") {
                            // 이전 버퍼 강제 비움 (stale 청크 방어)
                            {
                                std::lock_guard<std::mutex> lock(responses_stream_buffer_mutex);
                                responses_stream_buffer.clear();
                            }
                            is_responses_streaming = true;
                        }
                    }
                    std::lock_guard<std::mutex> lock(server_message_queue_mutex);
                    server_message_queue.push(response);
                    server_message_queue_cv.notify_one();
                }
            } catch (const json::parse_error& e) {
                std::cerr << "JSON 파싱 오류: " << e.what() << " | 원본 메시지: " << msg->str << std::endl;
            }
        }
        else if (msg->type == ix::WebSocketMessageType::Error) {
            std::cerr << "[WebSocket] 오류: " << msg->errorInfo.reason << std::endl;
        }
    });

    // 웹소켓 서버 시작 및 메인 루프
    auto res = ws_server.listen();
    if (!res.first) {
        std::cerr << "WebSocket 서버 listen 실패: " << res.second << std::endl;
        return -1;
    }
    ws_server.start();
    std::cout << "[WebSocket] 서버 시작됨 (port " << ws_port << "), Python 클라이언트 대기 중..." << std::endl;

    std::thread robot_thread(robot_main_loop, std::move(server_ready_future));
    robot_thread.join();
    ws_server.stop();
    g_ws_server = nullptr;
    ix::uninitNetSystem();

    #ifdef MOTOR_ENABLED
    if (tuning_logger) tuning_logger->stop();
    motion_logger.stop();
    cleanup_dynamixel();
    #endif

    return 0;
}