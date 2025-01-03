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
#include <deque>
#include <cmath>
#include <sndfile.h> // 오디오 파일 입출력을 위한 헤더
#include <fstream>
//#include <Egien/Dense>
// 매크로 및 필요한 함수 정의 포함
#include <cstdlib>
#include <ctime>
#include <algorithm>
#include <random>
#include <Eigen/Dense>
#include <unsupported/Eigen/Splines>
#include <tuple>
#include "cnpy.h"
#include "Macro_function.h"

// 파일 경로 상수
#define INPUT_FILE       "recorded_audio.wav"

#define INTERVAL_MS 360 // 시퀀스 1개 당 시간


#define RESULT_FILE "result_audio.wav" // 오디오 파일 경로
#define AGAIN_FILE "again.wav" // 중간에 말이 들어왔을 때 인터럽트 기능으로 다시 말해달라는 오디오


using namespace std;
using namespace Eigen;

// 전역 변수 및 동기화 도구
std::mutex mtx;
std::condition_variable cv;
std::chrono::time_point<std::chrono::high_resolution_clock> start_time;
std::atomic<bool> stop_flag(false);
std::atomic<bool> interrupt_flag(false);
std::atomic<bool> again_flag(false);

int first_move_flag = 1;
int first_run_flag = 1;
float final_result = 0.0f;
std::queue<std::vector<float>> audio_queue;
std::mutex audio_queue_mutex;
std::condition_variable audio_queue_cv;

bool use_first_output = true; // 파일 스위칭을 위한 플래그
std::mutex use_first_output_mutex;


std::queue<std::pair<int, float>> motion_queue; // 사이클 번호와 모션 값 저장"recorded_audio.wav"
std::queue<std::vector<std::vector<double>>> slice_queue; // 슬라이스 저장 및 전달 (headmotion)
std::mutex motion_queue_mutex;
std::condition_variable motion_queue_cv;

std::deque<float> recent_samples;
float sum_of_samples = 0.0f;

bool is_result_ready = false;

int DXL_goal_position[DXL_NUM] = {0, 0, 0, 0, 0};
int DXL_past_position[DXL_NUM] = {0, 0, 0, 0, 0};

// 로그 출력을 위한 뮤텍스
std::mutex cout_mutex;


std::string execPythonScript(const std::string& file_path) {
    std::string command = "python3 stt_script.py " + file_path;
    std::array<char, 128> buffer;
    std::string result;

    // popen을 사용해 Python 스크립트 호출 및 결과 가져오기
    std::shared_ptr<FILE> pipe(popen(command.c_str(), "r"), pclose);
    if (!pipe) throw std::runtime_error("popen() failed!");

    while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) {
        result += buffer.data();
    }
    return result;
}

std::string execGPTScript(const std::string& prompt) {
    std::string command = "python3 gpt_script.py \"" + prompt + "\"";
    std::array<char, 128> buffer;
    std::string result;

    std::shared_ptr<FILE> pipe(popen(command.c_str(), "r"), pclose);
    if (!pipe) throw std::runtime_error("popen() failed!");

    while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) {
        result += buffer.data();
    }
    return result;
}

std::string execTtsPythonScript(const std::string& text, const std::string& output_file) {
    std::string command = "python3 tts_script.py \"" + text + "\" " + output_file;
    std::array<char, 128> buffer;
    std::string result;

    std::shared_ptr<FILE> pipe(popen(command.c_str(), "r"), pclose);
    if (!pipe) throw std::runtime_error("popen() failed!");

    while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) {
        result += buffer.data();
    }
    return result;
}

std::string execGptTtsScript(const std::string& prompt, const std::string& output_file) {
    // 새로 합친 스크립트 사용
    std::string command = "python3 gpt_script.py \"" + prompt + "\" \"" + output_file + "\"";

    std::array<char, 128> buffer;
    std::string result;

    // 표준 출력(stdout)을 받아오기
    std::shared_ptr<FILE> pipe(popen(command.c_str(), "r"), pclose);
    if (!pipe) throw std::runtime_error("popen() failed!");

    while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) {
        result += buffer.data();
    }
    return result;  // 최종 GPT 답변 텍스트 (stdout로 찍힌 것)
}

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



class MySoundRecorder : public sf::SoundRecorder {
public:
    MySoundRecorder() : isSilent(true), start_flag(0) {
        setChannelCount(1);
        if (!sf::SoundRecorder::isAvailable()) {
            std::cerr << "Error: Audio capture is not supported on this device." << std::endl;
        }
    }

    bool onStart() override {
        std::cout << "Recording started..." << std::endl;
        audioSamples.clear();
        isSilent = true;
        start_flag = 0; // 녹음 시작 시 초기화
        start_flag_count = 0;
        return true;
    }

    bool onProcessSamples(const sf::Int16* samples, std::size_t sampleCount) override {
        audioSamples.insert(audioSamples.end(), samples, samples + sampleCount);

        bool silent = true;
        for (size_t i = 0; i < sampleCount; ++i) {
            if (std::abs(samples[i]) > silenceThreshold) { // 임계값 초과 시 음성 감지
                silent = false;
                if(start_flag_count > 5) start_flag = 1; // 음성이 일정 이상 들어오면 start_flag 설정
                else{
                    start_flag_count ++;
                }
                //std::cout << "start_flag = : " << start_flag << '\n';
                std::cout << "samples : " << abs(samples[i]) << '\n';
                break;
            }
        }
        isSilent = silent;
        return true;
    }

    void onStop() override {
        std::cout << "Recording stopped." << std::endl;
        saveToWav("recorded_audio.wav");
    }

    void saveToWav(const std::string& filename) {
        if (audioSamples.empty()) {
            std::cerr << "No audio samples to save." << std::endl;
            return;
        }

        sf::SoundBuffer buffer;
        if (!buffer.loadFromSamples(audioSamples.data(), audioSamples.size(), getChannelCount(), getSampleRate())) {
            std::cerr << "Error: Could not load samples into buffer." << std::endl;
            return;
        }

        if (!buffer.saveToFile(filename)) {
            std::cerr << "Error: Could not save audio to file." << std::endl;
        } else {
            std::cout << "Audio saved to " << filename << std::endl;
        }
    }

    bool isSilent; // 공백 상태 확인 플래그
    int start_flag; // 음성이 일정 이상 입력되었는지 확인하는 플래그
    int start_flag_count;

private:
    std::vector<sf::Int16> audioSamples;
    const int silenceThreshold = 3050; // 음성 인식 임계값
};



class AdaptiveSmoothFilter {
public:
    AdaptiveSmoothFilter(float alpha, int smooth_window) 
        : alpha(alpha), smooth_window(smooth_window) {}

    float apply(float input) {
        // 값이 갑자기 낮아진 경우를 감지하기 위한 임계값 설정
        float threshold = 0.15f;
        
        // 갑작스럽게 값이 낮아지면 그 값을 기록하고 해당 값을 유지
        if (!history.empty() && std::fabs(input - history.back()) > threshold && input < history.back()) {
            sudden_drop = input;
            smoothing = true;
            smooth_count = 0;
        }

        // 갑작스런 감소 이후의 값들은 천천히 낮아지도록 스무딩
        float output = input;
        if (smoothing && smooth_count < smooth_window) {
            output = alpha * input + (1 - alpha) * sudden_drop;
            smooth_count++;
        } else {
            smoothing = false; // 스무딩 창이 지나면 다시 원상 복귀
        }
        
        // 히스토리에 새로운 값 추가
        history.push_back(output);
        if (history.size() > smooth_window) {
            history.pop_front();
        }

        return output;
    }

private:
    float alpha;               // 스무스 변화에 대한 민감도
    int smooth_window;         // 스무딩 창 크기
    bool smoothing = false;    // 현재 스무딩 중인지 여부
    int smooth_count = 0;      // 스무딩 적용한 샘플 수
    float sudden_drop = 0.0f;  // 갑작스럽게 낮아진 값
    std::deque<float> history; // 히스토리 저장
};

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

class BackgroundSoundRecorder : public sf::SoundRecorder {
public:
    BackgroundSoundRecorder() {
        setChannelCount(1);
    }
    bool onStart() override {
        interrupt_flag = 0;
        interrupt_flag_count = 0;
        return true;
    }

    bool onProcessSamples(const sf::Int16* samples, std::size_t sampleCount) override {
        // 인터럽트 트리거 감지
        if (isSoundDetected(samples, sampleCount)) {
            
            if(interrupt_flag_count  >= 3) {
                interrupt_flag = 1;
                std::cout << "Interrupt flag ON " << '\n';
            }
            else interrupt_flag_count ++;
        }
        return true;
    }
    int interrupt_flag_count;

private:
    bool isSoundDetected(const sf::Int16* samples, std::size_t sampleCount) {
        for (size_t i = 0; i < sampleCount; ++i) {
            if (std::abs(samples[i]) > 3000) {  // 감지 임계값
                return true;
            }
        }
        return false;
    }
};


// 첫 번째 쓰레드: 음성 녹음 및 분할
void record_and_split(SNDFILE* sndfile, const SF_INFO& sfinfo, CustomSoundStream& soundStream) {

    int channels = sfinfo.channels;
    int samplerate = sfinfo.samplerate;
    int frames_per_interval = samplerate * INTERVAL_MS / 1000;


    sf_count_t total_frames = sfinfo.frames; // 파일의 총 프레임 수

    std::vector<float> audioBuffer(frames_per_interval * channels);

    sf_count_t position = 0;

    bool playback_started = false; // 재생 시작 여부를 추적하는 변수

    for (int cycle_num = 0; ; ++cycle_num) {
        if (interrupt_flag) {  // 외부 입력 감지 시 인터럽트
            std::cout << "[Cycle " << cycle_num << "] External input detected - Stopping playback and restarting recording\n";
            soundStream.stop();
            soundStream.clearBuffer();
            break;
        }
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

        {
            std::lock_guard<std::mutex> lock(audio_queue_mutex);
            audio_queue.push(audioBuffer);
        }
        audio_queue_cv.notify_one();

        {
            std::lock_guard<std::mutex> lock(cout_mutex);
            std::cout << "[Cycle " << cycle_num << " | " << get_time_str() << "] Recording and processing audio data" << '\n';
        }

        // **재생 시작 조건 확인 및 재생 시작**
        if (!playback_started && cycle_num >= 2) {
            soundStream.play();
            playback_started = true;
            {
                std::lock_guard<std::mutex> lock(cout_mutex);
                std::cout << "[Cycle " << cycle_num << " | " << get_time_str() << "] Starting audio playback" << '\n';
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
    // 파일 열기 (텍스트 파일에 쓰기)
    std::ofstream motion_log("generate_motion.txt");
    if (!motion_log.is_open()) {
        std::cerr << "Error opening generate_motion.txt for writing.\n";
        return;
    }

    vector<double> prevEndOneBefore = {0.0, 0.0, 0.0};
    vector<double> prevEnd = {0.1, 0.1, 0.1};
    vector<vector<double>> deliverSegment;
    std::array<double, 3> lastValues = {0.0, 0.0, 0.0}; // roll, pitch, yaw의 마지막 값 저장
    vector<double> boundaries = {0.01623224, 0.02907711, 0.04192197};

    int first_segment_flag = 1;

    for (int cycle_num = 1; ; ++cycle_num) {
        wait_for_next_cycle(cycle_num);
        if (stop_flag && audio_queue.empty()) {
            cout << "generate motion break ------------------------" << '\n';
            break;
        }
        if (interrupt_flag) {  // 외부 입력 감지 시 인터럽트
            std::cout << "[Cycle " << cycle_num << "] External input detected - Stopping generate motion \n";
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

        for(int i = 0; i < num_motion_updates; ++i){

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
                cout << "stop flag : " << stop_flag << ", audio queue size : " << audio_queue.size() << '\n';
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
                        s_max = max(s_max, v1_max[i]);
                    }
                }
                else{
                    for(float i = v1_max.size() - 100; i< v1_max.size(); i++){
                        s_max = max(s_max,v1_max[i]);
                    }
                }
            }

            double sc = 0.4/s_max;

            cout << "s_max : " << s_max << " sc: " << sc << '\n';
            cout << "raw_sample : " << max_sample << '\n';
            max_sample = sc  * max_sample;
            cout << "sc_sample : " << max_sample << '\n';


            // max_sample을 이전 5개값과 평균을 내서 평균값을 max_sample로 사용
            if(num_motion_size > 5) max_sample = update_final_result(moving_average_window, MOVING_AVERAGE_WINDOW_SIZE, max_sample);
            else{
                update_final_result(moving_average_window, MOVING_AVERAGE_WINDOW_SIZE, max_sample);
            }

            cout << "AVG_sample : " << max_sample << '\n';

            // max_sample 값이 min_open 값 이하일 때 하이퍼 탄젠트 적용해서 mouth 모션을 좀 더 역동적으로 만들어줌.
            if(num_motion_size > 5){
                if(max_sample > min_open) final_result = max_sample;
                else{
                    final_result = AM_fun(min_open,B, max_sample, ex_v1_max_sc_avg, exx_v1_max_sc_avg, lim_delta_r);
                }
            }else{
                max_sample = 0;
            }
            cout << "ex_v1_max_sc_avg : " << ex_v1_max_sc_avg << ", exx_v1_max_sc_avg : " << exx_v1_max_sc_avg << '\n';
            exx_v1_max_sc_avg = ex_v1_max_sc_avg;
            ex_v1_max_sc_avg = max_sample;

            cout << "final_result : " << final_result << '\n';
            float calculate_result = calculate_mouth(final_result, MAX_MOUTH, MIN_MOUTH);
            cout<< "calculate result : " << calculate_result << '\n';   

            motion_results.push_back(calculate_result);
            
            double rms_value = calculateRMS(channel_divided, 0, frames_per_update);
            energy.push_back(rms_value);

            if(i == num_motion_updates - 1) {
                
                //평균 기울기 값 계산
                avg_grad = getSegmentAverageGrad(energy, "one2one" , "abs");

                // 평균 기울기 값이 4개 class 중 어디에 해당하는지 판단 
                segClass = assignClassWith1DMiddleBoundary(avg_grad, boundaries);
                cout << "Assigned class : " << segClass << endl;
                string filePath;

                switch (segClass) {
                    case 0: filePath =  "segment_0.npy"; break;
                    case 1: filePath =  "segment_1.npy"; break;
                    case 2: filePath =  "segment_2.npy"; break;
                    case 3: filePath =  "segment_3.npy"; break;
                    default:
                        cerr << "Invalid segClass: " << segClass << endl;
                        break;
                }

                cnpy::NpyArray segment = cnpy::npy_load(filePath);
                cout << "npy load complete" << '\n';

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

        {
            std::lock_guard<std::mutex> lock(cout_mutex);
            std::cout << "[Cycle " << cycle_num << " | " << get_time_str() << "] Generating robot motion" << '\n';
        }

        {
            std::lock_guard<std::mutex> lock(motion_queue_mutex);
            for ( const auto& result : motion_results) {
                motion_queue.push(std::make_pair(cycle_num, result));
            }
            slice_queue.push(deliverSegment);
        }
        motion_queue_cv.notify_one();
    }
    // 파일 닫기
    motion_log.close();
}


void control_motor() {
    first_run_flag = 1;
    // 모터 초기 설정 코드
    while(!motion_queue.empty()) motion_queue.pop();
    while(!slice_queue.empty()) slice_queue.pop();
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
        moveDXLtoDesiredPosition(groupSyncWriteVelocity, groupSyncWritePosition, DXL_ID, dummy_positions, DXL_PROFILE_VELOCITY);
        first_run_flag = 0;
    }

    // 최초 값으로 모터를 움직임
   // moveDXLtoDesiredPosition(groupSyncWriteVelocity, groupSyncWritePosition, DXL_ID, DXL_initial_position, DXL_PROFILE_VELOCITY);

    // 최초 모터 이동 후 값을 업데이트
    for (int i = 0; i < DXL_NUM; i++) {
        DXL_past_position[i] = DXL_initial_position[i];
    }

    int cycle_num = 2; // cycle_num을 2부터 시작
    vector<vector<double>> current_motion_data(9, vector<double>(3, 0.0));

    for (;; cycle_num++) {
        
        wait_for_next_cycle(cycle_num);
        std::pair<int, float> motion_data;

        if(!slice_queue.empty()){
            current_motion_data = slice_queue.front(); // 슬라이스 데이터 가져오기
            slice_queue.pop();  
        }
        // 조건 변수 대기 시 조건식 사용
        if (interrupt_flag) {  // 외부 입력 감지 시 인터럽트
            std::cout << "[Cycle " << cycle_num << "] External input detected - Stopping control motor\n";
            break;
        }
        if (stop_flag && motion_queue.empty()) {
            cout << "control_motor break1 --------------------" << '\n';
            break;
        }
        int num_motor_updates = INTERVAL_MS / 40;

        std::vector<int> DXL_goal_position_vec;
        
        for (int i = 0; i < num_motor_updates; ++i) {
            //cout << "stop flag : " << stop_flag << " motion queue size : " << motion_queue.size() << '\n';
            {
                std::unique_lock<std::mutex> lock(motion_queue_mutex);
                
                if (stop_flag && motion_queue.empty()) {
                    cout << "motion queue size :  " << motion_queue.size() << ", control_motor break2 --------------------" << '\n';
                    break;
                }
                //cout << "cycle 에 들어옴 " << '\n';
                // 현재 사이클 번호에 해당하는 모션 값이 큐에 있을 때까지 대기
                // std::cout << "motion_queue front cycle: " << motion_queue.front().first 
                //  << ", current cycle_num: " << cycle_num - 1 << '\n';

                motion_queue_cv.wait(lock, [&] {
                    return (stop_flag && motion_queue.empty()) || (!motion_queue.empty() && motion_queue.front().first == cycle_num - 1);
                });
                
                // 모션 값 가져오기
                motion_data = motion_queue.front();
                motion_queue.pop();
                
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
                for (int i = 0; i < DXL_NUM; i++)
                    DXL_goal_position[i] = (DXL_past_position[i] + DXL_goal_position[i]) / 2;
            }


            groupSyncRead.clearParam(); // 파라미터 초기화

            // 모터를 목표 위치로 이동
            //cout << "모터 구동 " << '\n';
            moveDXLtoDesiredPosition(groupSyncWriteVelocity, groupSyncWritePosition, DXL_ID, DXL_goal_position, DXL_PROFILE_VELOCITY);
            {
            std::lock_guard<std::mutex> lock(cout_mutex);
            std::cout << "[Cycle " << cycle_num << " | " << get_time_str() << "] Controlling motor based on generated motion..." << '\n';
            }
            // 이전 위치 업데이트
            for (int i = 0; i < DXL_NUM; i++) {
                DXL_past_position[i] = DXL_goal_position[i];
            }

            // 필요한 경우 대기 시간 추가
            std::this_thread::sleep_for(std::chrono::milliseconds(40));
        }

        
    }
    moveDXLtoDesiredPosition(groupSyncWriteVelocity, groupSyncWritePosition, DXL_ID, DXL_initial_position, 1000);
}




int main() {
    // 반복 시작 전 사용자 안내
    std::cout << "Press 'q' to stop the program." << std::endl;

    while (true) {
        // 입력된 키가 'q'이면 루프 종료
        if (std::cin.rdbuf()->in_avail() && std::cin.peek() == 'q') {
            std::cout << "Program ending." << std::endl;
            break;
        }
        stop_flag = 0;                          //파일 끝에 들어오면 종료를 위한 플래그, while 문 첫 시작에 초기화
        again_flag = 0;
        MySoundRecorder recorder;
        SF_INFO sfinfo;
        SNDFILE* sndfile;
        BackgroundSoundRecorder bgRecorder;
        auto start = std::chrono::steady_clock::now();

        if (!interrupt_flag){
            if (recorder.start(44100)) {
                while (true) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(100));

                    // 2초 동안 음성이 없으면 녹음 종료
                    if (recorder.start_flag == 1 && recorder.isSilent) {
                        auto current = std::chrono::steady_clock::now();
                        auto silentDuration = std::chrono::duration_cast<std::chrono::seconds>(current - start);
                        
                        if (silentDuration.count() >= 2) {
                            recorder.stop();
                            break;
                        }
                    } else if (!recorder.isSilent) {
                        start = std::chrono::steady_clock::now(); // 음성이 감지되면 타이머 초기화
                    }
                }
                std::string transcription = execPythonScript(INPUT_FILE);

                std::cout << "Transcription: " << transcription << std::endl;
               
                // std::string gpt_result = execGPTScript(transcription);
                // std::cout << "GPT Answer : " << gpt_result << '\n';

                // std::string tts_result = execTtsPythonScript(gpt_result, RESULT_FILE);

                std::string final_text = execGptTtsScript(transcription, RESULT_FILE);

                // if (!tts_result.empty()) {
                //     std::cout << tts_result << " TTS generated audio saved to: " << RESULT_FILE << std::endl;
                // } else {
                //     std::cerr << "Error generating TTS audio." << std::endl;
                //     return -1;
                // }


                sndfile = sf_open(RESULT_FILE, SFM_READ, &sfinfo);
                if (!sndfile) {
                    std::cerr << "Error opening input file: " << RESULT_FILE << '\n';
                    return -1;
                }
            }
        }
        else if(interrupt_flag){

            std::cout << "Again sequnce start " << '\n';
            sndfile = sf_open(AGAIN_FILE, SFM_READ, &sfinfo);
            //std::this_thread::sleep_for(std::chrono::milliseconds(30));
            if (!sndfile) {
                std::cerr << "Error opening input file: " << AGAIN_FILE << '\n';
                return -1;
            }
            again_flag = 1;
            interrupt_flag = 0; // interrupt_flag가 켜져있으면 각 쓰레드 동작을 못하기 때문에 꺼야됨
        }

        start_time = std::chrono::high_resolution_clock::now();

        CustomSoundStream soundStream(sfinfo.channels, sfinfo.samplerate);
        
        

        if(!again_flag) {
            
            bgRecorder.start(44100);
            std::cout << "Background Record Start " << '\n';
        }

        std::thread t1(record_and_split, sndfile, sfinfo, std::ref(soundStream));
        std::thread t2(generate_motion, sndfile, sfinfo);
        std::thread t3(control_motor);

        t1.join();
        t2.join();
        t3.join();

        if(!again_flag) {
            bgRecorder.stop();
            std::cout << "Backgroun Record Stop" << '\n';
        }
        sf_close(sndfile);
    
    }
        
    return 0;
}


