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
#include "cnpy.h"
#include "Macro_function.h"

// WebSocket 및 JSON 관련 헤더
#include <ixwebsocket/IXNetSystem.h>
#include <ixwebsocket/IXWebSocket.h>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

#define INTERVAL_MS 360 // 시퀀스 1개 당 시간

// 파일 경로 설정
const std::string ASSETS_DIR = "assets";
const std::string DATA_DIR = "data";
const std::string MUSIC_DIR = ASSETS_DIR + "/audio/music";
const std::string VOCAL_DIR = ASSETS_DIR + "/audio/vocal";
const std::string SEGMENTS_DIR = DATA_DIR + "/segments";
const std::string IDLE_MOTION_FILE = ASSETS_DIR + "/motion/empty_10min.csv";

// 전역 변수 및 동기화 도구
std::string vocal_file_path;

std::chrono::time_point<std::chrono::high_resolution_clock> start_time; // 쓰레드 대기 시간 설정용
std::chrono::time_point<std::chrono::system_clock> startTime;   //프로세스 수행 타임 체크용
double STT_DONE_TIME = 0.0; // STT 완료 시간 (사용자 입력 완료 후 음성 출력까지의 시간 측정용)

std::atomic<bool> stop_flag(false);

int first_move_flag = 1;
int first_run_flag = 1;
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

// 로그 출력을 위한 뮤텍스
std::mutex cout_mutex;

std::string ray_mode = "sleep";
std::string wait_mode_flag = "off";
bool music_flag = 0;
bool playing_music_flag = 0;
bool print_cycle_log = 0; // 사이클 로그 출력 여부 (임시)

// WebSocket 관련 전역 객체
ix::WebSocket webSocket;
std::queue<json> serverMessageQueue;
std::mutex serverMessageQueueMutex;
std::condition_variable serverMessageQueueCV;
std::promise<void> serverReadyPromise;

// 시간 포맷터 함수
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

// 첫 번째 쓰레드: 음성 녹음 및 분할
void record_and_split(SNDFILE* sndfile, const SF_INFO& sfinfo, CustomSoundStream& soundStream) {
    
    SF_INFO vocal_sfinfo;
    SNDFILE* vocal_sndfile = nullptr;
    

    if (playing_music_flag) {
        // **Vocal 파일 열기**
        vocal_sndfile = sf_open(vocal_file_path.c_str(), SFM_READ, &vocal_sfinfo);
        if (!vocal_sndfile) {
            std::cerr << "Error opening vocal file: " << vocal_file_path << std::endl;
            return;
        }
    }

    int channels = sfinfo.channels;
    int samplerate = sfinfo.samplerate;
    int frames_per_interval = samplerate * INTERVAL_MS / 1000;


    sf_count_t total_frames = sfinfo.frames; // 파일의 총 프레임 수

    std::vector<float> audioBuffer(frames_per_interval * channels);
    std::vector<float> vocalBuffer(frames_per_interval * channels);

    sf_count_t position = 0;

    bool playback_started = false; // 재생 시작 여부를 추적하는 변수

    for (int cycle_num = 0; ; ++cycle_num) {

        wait_for_next_cycle(cycle_num);

        // 파일 위치 설정
        sf_seek(sndfile, position, SEEK_SET);

        // 남은 프레임 수 계산
        sf_count_t remaining_frames = total_frames - position;

        // 실제로 읽을 프레임 수 결정
        sf_count_t frames_to_read = std::min(remaining_frames, (sf_count_t)frames_per_interval);

        // 오디오 데이터 읽기
        sf_count_t framesRead = sf_readf_float(sndfile, &audioBuffer[0], frames_to_read);

        // 읽은 데이터가 없으면 종료
        if (framesRead == 0) break;

        // 읽은 데이터에 맞게 버퍼 크기 조정
        audioBuffer.resize(framesRead * channels);

        // float 데이터를 sf::Int16 타입으로 변환 -> sf::SoundStream 클래스는 sf::Int16 타입의 오디오 데이터를 사용 (16비트 : -32768 ~ 32767)
        std::vector<sf::Int16> int16Data(audioBuffer.size());
        for (std::size_t i = 0; i < audioBuffer.size(); ++i) {
            int16Data[i] = static_cast<sf::Int16>(audioBuffer[i] * 32767);
        }

        // 사운드 스트림에 데이터 추가
        soundStream.appendData(int16Data);

        if(playing_music_flag) {
            sf_count_t vocal_framesRead = sf_readf_float(vocal_sndfile, &vocalBuffer[0], frames_to_read);
            // 보컬 버퍼 크기 조정
            vocalBuffer.resize(vocal_framesRead * channels);
            {
                std::lock_guard<std::mutex> lock(audio_queue_mutex);
                audio_queue.push(vocalBuffer);
            }
            audio_queue_cv.notify_one();
        } else {
            {
                std::lock_guard<std::mutex> lock(audio_queue_mutex);
                audio_queue.push(audioBuffer);
            }
            audio_queue_cv.notify_one();
        }

        if (print_cycle_log) {
            std::lock_guard<std::mutex> lock(cout_mutex);
            std::cout << "[Cycle " << cycle_num << " | " << get_time_str() << "] Recording and processing audio data" << std::endl;
        }

        // **재생 시작 조건 확인 및 재생 시작**
        if (!playback_started && cycle_num >= 2) {
            soundStream.play();
            playback_started = true;
            std::chrono::duration<double, std::milli> playback_start_time = std::chrono::high_resolution_clock::now().time_since_epoch();
            {
                std::lock_guard<std::mutex> lock(cout_mutex);
                std::cout << "[Cycle " << cycle_num << " | " << get_time_str() << "] Starting audio playback" << "\n"
                          << "[시간 측정 2] 사용자 입력 완료 후 음성 재생까지의 시간: " << playback_start_time.count() - STT_DONE_TIME << "ms" << std::endl;
            }
        }

        // 다음 위치로 이동 (겹침을 고려하여 INTERVAL_MS 만큼 이동)
        position += frames_per_interval;

        // 파일의 끝에 도달하면 종료
        if (position >= total_frames) {
            stop_flag = true;
            break;
        }
    }
}


// 두 번째 쓰레드: 녹음본을 가지고 모션 생성
void generate_motion(SNDFILE* sndfile, const SF_INFO& sfinfo) {

    std::vector<float> audioBuffer;
    const size_t MOVING_AVERAGE_WINDOW_SIZE = 5;
    const size_t MAX_SAMPLE_WINDOW_SIZE = 40;
    std::deque<float> moving_average_window;

    while(!audio_queue.empty()) audio_queue.pop();      //인터럽트시 이미 차있는 오디오 큐 비어줘야함.
    
    int channels = sfinfo.channels;
    int samplerate = sfinfo.samplerate;

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
    std::vector<double> prevEnd = {0.1, 0.1, 0.1};
    std::vector<std::vector<double>> deliverSegment;
    std::array<double, 3> lastValues = {0.0, 0.0, 0.0}; // roll, pitch, yaw의 마지막 값 저장
    std::vector<double> boundaries = {0.01623224, 0.02907711, 0.04192197};

    int first_segment_flag = 1;

    for (int cycle_num = 1; ; ++cycle_num) {
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
        audio_queue_cv.wait(lock, [] {return !audio_queue.empty();});

        audioBuffer = std::move(audio_queue.front());
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
            if (end_frame > audioBuffer.size() / channels) {
                end_frame = audioBuffer.size() / channels;
                //i = num_motion_updates -1;
            }
            if (start_frame_mouth >= audioBuffer.size() / channels) {
                // 마지막 구간이므로 더 이상 처리할 오디오가 없음
                // break 또는 return 등을 통해 해당 루프/함수를 빠져나감
                std::cout << "stop flag : " << stop_flag << ", audio queue size : " << audio_queue.size() << std::endl;
                break; 
            }
            // 범위 체크
            if (end_frame_mouth > audioBuffer.size() / channels) {
                end_frame_mouth = audioBuffer.size() / channels;
                //i = num_motion_updates -1;
            }

            // 현재 업데이트에 해당하는 오디오 데이터 추출
            std::vector<float> current_audio(audioBuffer.begin() + start_frame * channels,
                                             audioBuffer.begin() + end_frame * channels);

            std::vector<float> current_audio_mouth(audioBuffer.begin() + start_frame_mouth * channels,
                                             audioBuffer.begin() + end_frame_mouth * channels);
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
            
            // -- 헤드 모션 생성 --
            double rms_value = calculateRMS(channel_divided, 0, frames_per_update);
            energy.push_back(rms_value);

            if(i == num_motion_updates - 1) {
                
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
                //cout << "npy load complete" << '\n';

                //segment 선택
                deliverSegment = getNextSegment_SegSeg(prevEndOneBefore, prevEnd, segment, true, true);
            
                // segment 보정 (무성구간에 따라서 값 보정)
                deliverSegment = multExpToSegment(energy, deliverSegment, 0.01, 10);

                // 이전 Segment의 끝부분과 현재 Segment의 시작부분을 B-spline 보간법을 통해 자연스럽게 이어줌.
                if(first_segment_flag == 1) first_segment_flag = 0;
                else{
                   deliverSegment = connectTwoSegments(prevEndOneBefore, lastValues,deliverSegment,3);
                }

                for(int p = 0; p < 3; p++){
                    lastValues[p] = deliverSegment.back()[p];
                }
                // 여기 부분 lastValues 쓸껀지, PrevEnd 쓸껀지 선택해야할듯 
                for(int j = 0; j < 3; j++){
                    prevEnd[j] = deliverSegment[8][j]; // deliverSegment의 마지막 데이터 값
                }
                for(int j = 0; j < 3; j++){
                    prevEndOneBefore[j] = deliverSegment[7][j];
                }
            }
        }

        if (print_cycle_log) {
            std::lock_guard<std::mutex> lock(cout_mutex);
            std::cout << "[Cycle " << cycle_num << " | " << get_time_str() << "] Generating robot motion" << std::endl;
        }

        {
            std::lock_guard<std::mutex> lock(mouth_motion_queue_mutex);
            for ( const auto& result : motion_results) {
                mouth_motion_queue.push(std::make_pair(cycle_num, result));
            }
            head_motion_queue.push(deliverSegment);
        }
        mouth_motion_queue_cv.notify_one();
    }
}


void control_motor() {
    first_run_flag = 1;
    // 모터 초기 설정 코드
    while(!mouth_motion_queue.empty()) mouth_motion_queue.pop();
    while(!head_motion_queue.empty()) head_motion_queue.pop();
    dynamixel::PortHandler *portHandler = dynamixel::PortHandler::getPortHandler(DEVICENAME);
    dynamixel::PacketHandler *packetHandler = dynamixel::PacketHandler::getPacketHandler(PROTOCOL_VERSION);

    // GroupSyncWrite 인스턴스 초기화
    dynamixel::GroupSyncWrite groupSyncWritePosition(portHandler, packetHandler, ADDR_PRO_GOAL_POSITION, LEN_PRO_GOAL_POSITION);
    dynamixel::GroupSyncWrite groupSyncWriteVelocity(portHandler, packetHandler, ADDR_PRO_PROFILE_VELOCITY, LEN_PRO_PROFILE_VELOCITY);
    // Present Position을 위한 GroupSyncRead 인스턴스 초기화
    dynamixel::GroupSyncRead groupSyncRead(portHandler, packetHandler, ADDR_PRO_PRESENT_POSITION, LEN_PRO_PRESENT_POSITION);

    int DXL_ID[DXL_NUM] = { DXL1_ID, DXL2_ID, DXL3_ID, DXL4_ID, DXL5_ID };
    int DXL_initial_position[DXL_NUM] = { DEFAULT_PITCH, DEFAULT_ROLL_R, DEFAULT_ROLL_L, DEFAULT_YAW, DEFAULT_MOUTH };
    uint8_t dxl_error = 0;
    
    if (!(portHandler->openPort())) {
        printf("Failed to open the port!\n");
        return;
    }

    // 최초 모터 이동 후 값을 업데이트
    for (int i = 0; i < DXL_NUM; i++) {
        DXL_past_position[i] = DXL_initial_position[i];
    }

    int cycle_num = 2; // cycle_num을 2부터 시작
    std::vector<std::vector<double>> current_motion_data(9, std::vector<double>(3, 0.0));

    for (;; cycle_num++) {
        
        wait_for_next_cycle(cycle_num);
        std::pair<int, float> motion_data;

        if(!head_motion_queue.empty()){
            current_motion_data = head_motion_queue.front(); // 슬라이스 데이터 가져오기
            head_motion_queue.pop();  
        }
        
        if (stop_flag && mouth_motion_queue.empty()) {
            std::cout << "control_motor break1 --------------------" << std::endl;
            break;
        }
        int num_motor_updates = INTERVAL_MS / 40;

        std::vector<int> DXL_goal_position_vec;
        
        for (int i = 0; i < num_motor_updates; ++i) {
            //cout << "stop flag : " << stop_flag << " motion queue size : " << mouth_motion_queue.size() << '\n';
            {
                std::unique_lock<std::mutex> lock(mouth_motion_queue_mutex);
                
                if (stop_flag && mouth_motion_queue.empty()) {
                    std::cout << "motion queue size :  " << mouth_motion_queue.size() << ", control_motor break2 --------------------" << std::endl;
                    break;
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


            groupSyncRead.clearParam(); // 파라미터 초기화

            // 모터를 목표 위치로 이동
            //cout << "모터 구동 " << '\n';
            moveDXLtoDesiredPosition(groupSyncWriteVelocity, groupSyncWritePosition, DXL_ID, DXL_goal_position, DXL_PROFILE_VELOCITY);
            
            // 이전 위치 업데이트
            for (int i = 0; i < DXL_NUM; i++) {
                DXL_past_position[i] = DXL_goal_position[i];
            }
            if (print_cycle_log) {
                std::lock_guard<std::mutex> lock(cout_mutex);
                std::cout << "[Cycle " << cycle_num << " | " << get_time_str() << "] Controlling motor based on generated motion..." << std::endl;
            }
            // 필요한 경우 대기 시간 추가
            std::this_thread::sleep_for(std::chrono::milliseconds(39));
        }

        
    }
    moveDXLtoDesiredPosition(groupSyncWriteVelocity, groupSyncWritePosition, DXL_ID, DXL_initial_position, 1000);
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

void initDynamixel(){
    
}
void wait_control_motor(){
    first_run_flag = 1;
    // 모터 초기 설정 코드
    std::cout << "ray mode : " << ray_mode << std::endl;
    if(wait_mode_flag == "off") return;
    while(!mouth_motion_queue.empty()) mouth_motion_queue.pop();
    while(!head_motion_queue.empty()) head_motion_queue.pop();
    dynamixel::PortHandler *portHandler = dynamixel::PortHandler::getPortHandler(DEVICENAME);
    dynamixel::PacketHandler *packetHandler = dynamixel::PacketHandler::getPacketHandler(PROTOCOL_VERSION);

    // GroupSyncWrite 인스턴스 초기화
    dynamixel::GroupSyncWrite groupSyncWritePosition(portHandler, packetHandler, ADDR_PRO_GOAL_POSITION, LEN_PRO_GOAL_POSITION);
    dynamixel::GroupSyncWrite groupSyncWriteVelocity(portHandler, packetHandler, ADDR_PRO_PROFILE_VELOCITY, LEN_PRO_PROFILE_VELOCITY);
    // Present Position을 위한 GroupSyncRead 인스턴스 초기화
    dynamixel::GroupSyncRead groupSyncRead(portHandler, packetHandler, ADDR_PRO_PRESENT_POSITION, LEN_PRO_PRESENT_POSITION);

    int DXL_ID[DXL_NUM] = { DXL1_ID, DXL2_ID, DXL3_ID, DXL4_ID, DXL5_ID };
    int DXL_initial_position[DXL_NUM] = { DEFAULT_PITCH, DEFAULT_ROLL_R, DEFAULT_ROLL_L, DEFAULT_YAW, DEFAULT_MOUTH };
    uint8_t dxl_error = 0;


    if (first_run_flag == 1) {
        if (!(portHandler->openPort())) {
            printf("Failed to open the port!\n");
            return;
        }

        if (!(portHandler->setBaudRate(BAUDRATE))) {
            printf("Failed to change the baudrate!\n");
            return;
        }

        if (!enable_torque(packetHandler, portHandler, DXL_ID, dxl_error)) {
            printf("Failed to torque enable\n");
            return;
        }
        int dummy_positions[DXL_NUM] = { DEFAULT_PITCH, DEFAULT_ROLL_R, DEFAULT_ROLL_L, DEFAULT_YAW, DEFAULT_MOUTH };

        //cout << "첫번째 모터 구동 " << '\n';
        moveDXLtoDesiredPosition(groupSyncWriteVelocity, groupSyncWritePosition, DXL_ID, dummy_positions, 300);
        first_run_flag = 0;
    }

   std::this_thread::sleep_for(std::chrono::milliseconds(200));

    constexpr auto FRAME_INTERVAL = std::chrono::milliseconds(35);
    std::string line;
    while(wait_mode_flag == "on"){
        std::ifstream headGesture(IDLE_MOTION_FILE);
        if (!headGesture) {
            std::cerr << "Empty HeadGesture File not found." << std::endl;
            return;
        }
        while(headGesture.good()){
            auto headRow = csv_read_row(headGesture, ',');
            if(wait_mode_flag == "off") break;
            float roll_s = std::stof(headRow[0]);
            float pitch_s = std::stof(headRow[1]);
            float yaw_s = std::stof(headRow[2]);
            float mouth_s = 0;

            double ratiooo = 1.4;

            std::vector<int> DXL = RPY2DXL(roll_s * ratiooo , pitch_s *ratiooo, yaw_s * ratiooo, mouth_s, 0);
            
            update_DXL_goal_position(DXL_goal_position,
                                        DXL[0],
                                        DXL[1],
                                        DXL[2],
                                        DXL[3],
                                        DXL[4]);
        

            groupSyncRead.clearParam(); // 파라미터 초기화

            moveDXLtoDesiredPosition(groupSyncWriteVelocity, groupSyncWritePosition, DXL_ID, DXL_goal_position, DXL_PROFILE_VELOCITY);

            std::this_thread::sleep_for(FRAME_INTERVAL);
        }
    }
    std::cout << "wait mode finish " << std::endl;
   // portHandler -> closePort(); // 이거 계속 쓸꺼면 control_motor 함수에도 추가해주기 
}

void robot_main_loop(std::future<void> serverReadyFuture) {
    std::cout << "서버 연결 대기 중..." << std::endl;
    serverReadyFuture.get(); // 서버가 준비될 때까지 대기
    std::cout << "서버 연결 완료!" << std::endl;

    std::pair<std::string,std::string> play_music;
    while (true) {
        // 초기화
        stop_flag = 0; // 오디오 파일 끝에 도달 시 종료를 위한 플래그
        SF_INFO sfinfo;
        SNDFILE* sndfile;
        startTime = std::chrono::system_clock::now();
        std::chrono::duration<double, std::milli> START_TIME = std::chrono::high_resolution_clock::now().time_since_epoch();

        if(music_flag) {
            music_flag = 0;
            std::cout << "music_flag IN" << std::endl;
            std::string play_song_path = MUSIC_DIR + "/" + play_music.first + "_" + play_music.second + ".wav";
            vocal_file_path = VOCAL_DIR + "/" + play_music.first + "_" + play_music.second + "_" + "vocals" + ".wav";
            sndfile = sf_open(play_song_path.c_str(), SFM_READ, &sfinfo);
            std::cout << "------------------------------MUSIC ON ------------------------" << std::endl;
            playing_music_flag = 1;
        }
        else {
            // 대기모드 시작
            if (wait_mode_flag == "off") {
                wait_mode_flag = "on";
                std::thread wait_mode(wait_control_motor);
                wait_mode.detach();
            }

            // 서버에 다음 행동 요청
            json request;
            request["request"] = "next_action";
            webSocket.send(request.dump());
            
            // 서버로부터 응답 대기
            json response;
            {
                std::unique_lock<std::mutex> lock(serverMessageQueueMutex);
                if (!serverMessageQueueCV.wait_for(lock, std::chrono::seconds(30), [] { return !serverMessageQueue.empty(); })) {
                    std::cerr << "서버 응답 대기 시간 초과" << std::endl;
                    continue;
                }
                response = serverMessageQueue.front();
                serverMessageQueue.pop();
            }
            
            // 시간 측정
            double STT_READY_TIME = response.value("stt_ready_time", 0.0);
            STT_DONE_TIME = response.value("stt_done_time", 0.0);
            double GPT_RESPONSE_TIME = response.value("gpt_response_time", 0.0);
            std::string user_text = response.value("user_text", "");
            std::string gpt_response_text = response.value("gpt_response_text", "");
            std::cout << "[시간 측정 1] 사용자 입력 준비 시간: " << STT_READY_TIME - START_TIME.count() << "ms" << std::endl;
            std::cout << "[시간 측정] GPT 응답 시간: " << GPT_RESPONSE_TIME << "ms" << "\n      사용자 입력: " << user_text << "\n      GPT 응답: " << gpt_response_text << std::endl;
            
            // 서버 응답 처리
            std::string action = response.value("action", "error");
            if (action == "play_audio") {
                std::string file_to_play = response.value("file_to_play", "");
                sndfile = sf_open(file_to_play.c_str(), SFM_READ, &sfinfo);
            }
            else if (action == "play_music") {
                music_flag = 1;
                std::string file_to_play = response.value("file_to_play", "");
                std::string title = response.value("title", "");
                std::string artist = response.value("artist", "");
                play_music = std::make_pair(title, artist);
                sndfile = sf_open(file_to_play.c_str(), SFM_READ, &sfinfo);
            }
            else if (action == "sleep") {
                continue;
            }

            // 대기 모드 종료
            wait_mode_flag = "off";
        }

        if (!sndfile) {
            std::cerr << "Error opening audio file: " << sf_strerror(sndfile) << std::endl;
            continue;
        }
        CustomSoundStream soundStream(sfinfo.channels, sfinfo.samplerate);

        auto endTime = std::chrono::system_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::seconds>(endTime - startTime);
        std::cout << "응답 처리 시간: " << duration.count() << "초" << std::endl;

        start_time = std::chrono::high_resolution_clock::now();
        std::thread t1(record_and_split, sndfile, sfinfo, std::ref(soundStream));
        std::thread t2(generate_motion, sndfile, sfinfo);
        std::thread t3(control_motor);

        t1.join();
        t2.join();
        t3.join();

        sf_close(sndfile);
        playing_music_flag = 0;
    }
}

int main() {
    // 웹소켓 서버 준비
    std::future<void> serverReadyFuture = serverReadyPromise.get_future();
    ix::initNetSystem();
    webSocket.setUrl("ws://127.0.0.1:5000");
    webSocket.setOnMessageCallback([](const ix::WebSocketMessagePtr& msg) {
        if (msg->type == ix::WebSocketMessageType::Message) {
            try {
                json response = json::parse(msg->str);
                {
                    std::lock_guard<std::mutex> lock(serverMessageQueueMutex);
                    serverMessageQueue.push(response);
                }
                serverMessageQueueCV.notify_one();
            } catch (const json::parse_error& e) {
                std::lock_guard<std::mutex> lock(cout_mutex);
                std::cerr << "JSON 파싱 오류: " << e.what() << " | 원본 메시지: " << msg->str << std::endl;
            }
        } else if (msg->type == ix::WebSocketMessageType::Open) {
            std::lock_guard<std::mutex> lock(cout_mutex);
            std::cout << "[WebSocket] 서버에 성공적으로 연결되었습니다." << std::endl;
            serverReadyPromise.set_value(); // 서버가 준비되었음을 알림
        } else if (msg->type == ix::WebSocketMessageType::Error) {
            std::lock_guard<std::mutex> lock(cout_mutex);
            std::cerr << "[WebSocket] 연결 오류: " << msg->errorInfo.reason << std::endl;
        }
    });

    // 웹소켓 서버 및 메인 루프 시작
    webSocket.start();
    std::thread robotThread(robot_main_loop, std::move(serverReadyFuture));
    robotThread.join();
    webSocket.stop();
    ix::uninitNetSystem();
    return 0;
}