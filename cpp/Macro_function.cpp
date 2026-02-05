#include "Macro_function.h"
#include <algorithm>  // std::max, std::min

#include "Macro_function.h"
// 다른 include 들 ...

// =======================
//  순수 Attack-Release Envelope 구현
// =======================

// =======================
//  순수 Attack-Release Envelope 구현
// =======================

// 순수 AR 초기화 (실제 로직)
//   fs         : 샘플레이트 [Hz]
//   attack_ms  : Attack 시간 [ms]
//   release_ms : Release 시간 [ms]
void initMouthEnvAR(MouthEnvARState& st,
                    double fs,
                    double attack_ms,
                    double release_ms)
{
    if (fs <= 0.0)
        fs = 24000.0;  // 기본값 방어

    if (attack_ms  <= 0.0) attack_ms  = 1.0;
    if (release_ms <= 0.0) release_ms = 1.0;

    double attack_T  = attack_ms  * 0.001; // [s]
    double release_T = release_ms * 0.001; // [s]

    // a = exp(-1/(τ fs)) → env_new = a * env_old + (1 - a) * x
    double a_att = std::exp(-1.0 / (attack_T  * fs));
    double a_rel = std::exp(-1.0 / (release_T * fs));

    st.attack_a  = static_cast<float>(a_att);
    st.release_a = static_cast<float>(a_rel);
    st.env       = 0.0f;
}

// 한 샘플에 대해 Attack-Release 한 스텝 진행
//   x_in : 입력 음성 샘플 (-1~1 가정)
//   return : env 값
float processMouthEnvAR(MouthEnvARState& st, float x_in)
{
    // MATLAB처럼 abs(x) 기준으로 envelope 계산
    float x = std::fabs(x_in);

    if (x > st.env)
    {
        // Attack: 빠르게 따라감
        // env_new = attack_a * env_old + (1 - attack_a) * x
        st.env = st.attack_a * st.env + (1.0f - st.attack_a) * x;
    }
    else
    {
        // Release: 천천히 떨어짐
        // env_new = release_a * env_old + (1 - release_a) * x
        st.env = st.release_a * st.env + (1.0f - st.release_a) * x;
    }

    return st.env;
}



// env: 0~1 엔벨롭 (0=완전 닫힘, 1=최대 벌림)
// max_MOUTH: (지금은 안 씀, 시그니처 맞추려고 남겨둠)
// min_MOUTH: "최대 이동량" (예: 550틱)
#include <cmath>

#include <cmath>

// env      : AR 엔벨롭 결과 (vocal / vocals 공통)
// max_MOUTH: 지금은 안 씀 (시그니처 유지용)
// min_MOUTH: 최대 당김량 틱 (예: 550)  → rope를 이만큼 당기면 최대 벌림
float calculate_mouth(float env, float max_MOUTH, float min_MOUTH)
{
    // 1) AR 결과 절댓값
    float x = std::fabs(env);

    // 완전 무음 근처는 바로 0 (노이즈 방지)
    const float EPS = 1e-7f;
    if (x < EPS)
        return 0.0f;

    // 2) 이 스트림의 "평균적인 엔벨롭 크기" 추적 (AGC 느낌)
    static bool  initialized = false;
    static float ref_level   = 0.0f;   // 평균 레벨
    static float mouth_state = 0.0f;   // 입 상태(0~1), 여닫기 속도 제어용

    if (!initialized)
    {
        ref_level   = (x > 1e-4f) ? x : 1e-4f;
        mouth_state = 0.0f;
        initialized = true;
    }

    // 🔹 ref_level: 느리게 env를 따라가는 평균 (TTS/노래 둘 다 자동 적응)
    // 40 ms 주기 기준, alpha_ref=0.01 → 대략 1~2초 단위로 평균 갱신
    const float alpha_ref = 0.01f;
    ref_level = (1.0f - alpha_ref) * ref_level + alpha_ref * x;
    if (ref_level < 1e-4f) ref_level = 1e-4f;

    // 3) 평균 대비 현재 크기 (0 ~ 여러 배)
    float rel = x / ref_level;

    // 4) rel을 0~1 사이로 부드럽게 압축
    //    rel=0 → 0, rel=1 → 0.5, rel→∞ → 1
    float env_norm = rel / (1.0f + rel);   // 0~1

    // 5) "진짜 조용한 구간"에서는 변화량이 있어도 입 거의 안 벌리기
    //    quiet_ratio: ref_level 대비 어느 정도 이하여야 "조용"으로 볼지
    float ratio = x / ref_level;           // 0~∞
    const float quiet_ratio = 0.4f;        // ref_level의 40% 미만 → 조용한 영역
    float quiet_weight = (ratio - quiet_ratio) / (1.0f - quiet_ratio);
    if (quiet_weight < 0.0f) quiet_weight = 0.0f;
    if (quiet_weight > 1.0f) quiet_weight = 1.0f;

    // 조용한 구간에서는 env_norm이 더 줄어들게 만들기
    env_norm *= quiet_weight;              // 작은 소리 + 변화량 → 과민 반응 방지

    // 6) 사람 입 느낌 곡선 (gamma): 중간 구간 조금 더 살리기
    const float gamma = 0.9f;              // 1.0이면 직선, <1이면 중간 강조
    if (env_norm > 0.0f)
        env_norm = std::pow(env_norm, gamma);

    // 7) 여닫기 속도 제어 (mouth_state)
    //    - target: 이번 프레임에서 "이 정도 열려야 한다" (0~1)
    //    - mouth_state: 실제 모터에 적용할 상태 (0~1), 여닫기 속도 분리
    float target = env_norm;

   // 🔹 기본 속도
    float open_alpha_base  = 0.5f;   // 벌릴 때 기본 속도 (조금 더 빠르게)
    float close_alpha_base = 0.7f;   // 닫을 때 기본 속도 (조금 더 빠르게)

    // 🔹 추가: "입 꽉 닫기" 트리거
    //    target가 아주 낮으면(= 거의 소리 없음) 확실히 닫히도록 더 세게 끌어내리기
    const float close_hard_threshold = 0.2f;  // target이 0.2보다 작으면 "닫아야 하는" 구간
    const float close_hard_extra     = 0.3f;  // 추가로 더 빠르게 닫는 비율

    if (target > mouth_state)
    {
        // 입 벌리기: 조금 더 빠르게, 하지만 너무 튀지 않게
        float alpha = open_alpha_base;
        mouth_state += alpha * (target - mouth_state);
    }
    else
    {
        // 입 닫기: 기본은 빨리 닫되,
        // target이 아주 작으면(거의 무음) 더 과감하게 닫아 버림
        float alpha = close_alpha_base;

        if (target < close_hard_threshold)
        {
            alpha += close_hard_extra;  // 예: 0.6 + 0.3 = 0.9
            if (alpha > 1.0f) alpha = 1.0f; // 안전
        }

        mouth_state += alpha * (target - mouth_state);
    }
    // 안전 클램프
    if (mouth_state < 0.0f) mouth_state = 0.0f;
    if (mouth_state > 1.0f) mouth_state = 1.0f;

    // 8) 기구 한계 보호: 최대 오픈을 90%까지만 사용 (풀리 끝까지 안 가게)
    const float max_open_ratio = 0.9f;
    float final_norm = mouth_state * max_open_ratio;

    // 9) 최종: 0 ~ min_MOUTH 틱으로 변환 (rope 당김량)
    return final_norm * min_MOUTH;
}




//파일 경로 생성
void create_file_path(std::string &file_path, const char *filename, const char *input_path) 
{
  std::stringstream ss;
  ss << input_path;
  if(filename != nullptr)
  {
    ss << filename;
  }

  file_path = ss.str();
}
void save_one_float_to_csv(const std::string& filename, float value) {
    std::ofstream file(filename, std::ios::app); // append 모드로 파일 열기

    if (!file.is_open()) {
        std::cerr << "Error: Unable to open file " << filename << std::endl;
        return;
    }

    file << value << '\n'; // float 값을 파일에 쓰기
    file.close();
    std::cout << "Float value saved to " << filename << " successfully." << std::endl;
}

void save_to_csv_float(const std::string& base_filename, const std::vector<float>& data, int frame_count) {

    std::ostringstream filename;
    filename << base_filename << "_" << frame_count << ".csv";
    std::ofstream file(filename.str());

    if (!file.is_open()) {
        std::cerr << "Error: Unable to open file " << filename.str() << std::endl;
        return;
    }

    for (size_t i = 0; i < data.size(); ++i) {
        file << data[i] <<'\n';
    }
    file << std::endl; // 각 행을 다음 줄로 나눔

    file.close();
    std::cout << "Data saved to " << filename.str() << " successfully." << std::endl;
}

void save_audio_file(const std::string& filename, const float* audiodata, sf_count_t frames, int samplerate, int channels) {
    SF_INFO sfinfo;
    sfinfo.channels = channels;
    sfinfo.samplerate = samplerate;
    sfinfo.format = SF_FORMAT_WAV | SF_FORMAT_PCM_16;

    SNDFILE* outfile = sf_open(filename.c_str(), SFM_WRITE, &sfinfo);
    if (!outfile) {
        std::cerr << "Error: Unable to open output file " << filename << std::endl;
        return;
    }

    sf_write_float(outfile, audiodata, frames * channels);
    sf_close(outfile);
}


std::pair<float, size_t> find_peak(const std::vector<float>& audio_buffer) {
    float max_sample = audio_buffer[0];
    size_t max_index = 0;

    for (size_t i = 1; i < audio_buffer.size(); ++i) {
        if (audio_buffer[i] > max_sample) {
            max_sample = audio_buffer[i];
            max_index = i;
        }
    }

    return {max_sample, max_index};
}


void save_audio_segment(const std::string& outputFilePath, const std::vector<float>& audioData, size_t dataSize) {
    // Open the output file
    SF_INFO sfinfo;
    sfinfo.channels = 1;
    sfinfo.samplerate = 44100;
    sfinfo.format = SF_FORMAT_WAV | SF_FORMAT_FLOAT;
    SNDFILE *outfile = sf_open(outputFilePath.c_str(), SFM_WRITE, &sfinfo);
    if (!outfile) {
        std::cerr << "Error opening output file: " << outputFilePath << std::endl;
        return;
    }

    // Write audio data to the output file
    sf_write_float(outfile, audioData.data(), dataSize);

    // Close the output file
    sf_close(outfile);

    std::cout << "Audio segment saved to file: " << outputFilePath << std::endl;
}

// 채널 데이터를 분리하는 함수
std::vector<float> divide_channel(const std::vector<float>& audio_data, int channels, int frames) {
    if (channels == 1)
    {
        // 단일 채널의 경우 원본 데이터를 그대로 반환
        return audio_data;
    }
    else if (channels == 2)
    {
        // 채널을 분리해주기 위한 벡터 생성
        std::vector<float> left_channel_data(frames);
        std::vector<float> right_channel_data(frames);

        // 각 채널의 데이터를 분리하여 저장
        for (int i = 0; i < frames; ++i)
        {
            left_channel_data[i] = audio_data[2 * i];       // 짝수 번째 인덱스는 왼쪽(L) 채널의 데이터
            right_channel_data[i] = audio_data[2 * i + 1];  // 홀수 번째 인덱스는 오른쪽(R) 채널의 데이터
        }

        return left_channel_data;
    }
    else
    {
        // 지원되지 않는 채널 수
        std::cerr << "지원되지 않는 채널 수입니다: " << channels << std::endl;
        return std::vector<float>();
    }
}

float moving_average(const std::deque<float>& window) {
    if (window.empty()) return 0.0f;
    float sum = std::accumulate(window.begin(), window.end(), 0.0f);
    return sum / window.size();
}

//이동 평균 윈도우를 업데이트하고 최종 결과를 반환하는 함수
float update_final_result(std::deque<float>& moving_average_window, size_t window_size, float new_sample) {
    moving_average_window.push_back(new_sample);

    if (moving_average_window.size() < window_size) {
        // 이동 평균 윈도우 크기가 설정된 크기보다 작은 경우에는 단순히 새 샘플을 반환
        return new_sample;
    } else {
        // 이동 평균 윈도우 크기가 설정된 크기보다 크거나 같은 경우에는 이동 평균을 계산하고 그 결과를 반환
        float moving_result = moving_average(moving_average_window);
        // 이동 평균 윈도우에서 가장 오래된 샘플 제거
        moving_average_window.pop_front();
        return moving_result;
    }
}



float scale_max_sample(float max_sample) {
    // max_sample이 0 ~ 1 범위에 있을 때
    float scaling_factor = 1.5f;  // atan 함수 스케일링 인자
    float n = 2.0f;               // 거듭제곱 지수 (n > 1)

    if (max_sample <= 0.15f) {
        // 0.15 이하일 때는 원래 값 그대로 반환
        float result = std::atan(scaling_factor * max_sample) / std::atan(scaling_factor);
        return result;
    } else if (max_sample >= 0.25f) {
        // 0.25 이상일 때는 변환 함수 적용
        float result = std::atan(scaling_factor * max_sample) / std::atan(scaling_factor);
        float adjusted_result = std::pow(result, n);
        return adjusted_result;
    } else {
        // 0.15 < max_sample < 0.25 사이에서는 선형 보간 적용
        // 변환된 값 계산
        float result = std::atan(scaling_factor * max_sample) / std::atan(scaling_factor);
        float adjusted_result = std::pow(result, n);

        // 비율 계산 (0.15에서 0, 0.25에서 1)
        float t = (max_sample - 0.15f) / (0.25f - 0.15f);

        // 선형 보간하여 adjusted_result와 max_sample 사이를 연결
        float interpolated_result = (1.0f - t) * max_sample + t * adjusted_result;

        return interpolated_result;
    }
}


float scaled_result_with_moving_average(std::deque<float>& recent_samples, size_t window_size, float new_sample) {
    // 새로운 샘플을 이동 평균 윈도우에 추가
    recent_samples.push_back(new_sample);

    // 샘플의 합을 계산
    float sum_of_samples = std::accumulate(recent_samples.begin(), recent_samples.end(), 0.0f);

    // 윈도우 크기가 설정된 크기보다 작을 때는 평균을 그대로 계산
    float average_sample;
    if (recent_samples.size() < window_size) {
        average_sample = sum_of_samples / recent_samples.size();
    } else {
        // 윈도우 크기가 설정된 크기와 같거나 클 때는 평균 계산 후 이동
        average_sample = sum_of_samples / window_size;
        // 가장 오래된 샘플을 윈도우에서 제거하여 이동 평균 업데이트
        recent_samples.pop_front();
    }

    // 비율을 이용한 스케일링 계산
    float scaled_result = std::atan(new_sample) / std::atan( average_sample);


    return scaled_result * 0.2f;  // 최종적으로 0.3 스케일 적용
}

float volume_control(std::deque<float>& recent_samples, size_t window_size, float new_sample){
    // 새로운 샘플을 이동 평균 윈도우에 추가
    
     float scaled_result;

     recent_samples.push_back(new_sample);
    // 샘플의 합을 계산
    float sum_of_samples = std::accumulate(recent_samples.begin(), recent_samples.end(), 0.0f);

    // 윈도우 크기가 설정된 크기보다 작을 때는 평균을 그대로 계산
    float average_sample;
    if (recent_samples.size() < window_size) {
        average_sample = sum_of_samples / recent_samples.size();
    } else {
        // 윈도우 크기가 설정된 크기와 같거나 클 때는 평균 계산 후 이동
        average_sample = sum_of_samples / window_size;
        // 가장 오래된 샘플을 윈도우에서 제거하여 이동 평균 업데이트
        recent_samples.pop_front();
    }
    if(average_sample > 0.6){
       // std::cout << "average_sample : " << average_sample << " new sample : " << new_sample << " pow : " << pow(new_sample,1.7) << '\n';
        scaled_result = (2*COMPENSATION_MAX/PI) *std::atan(PI*INCLINATION*pow(new_sample,1.9)/(2*COMPENSATION_MAX));
    }
    else if(average_sample > 0.4){
        //std::cout << "average_sample : " << average_sample << " new sample : " << new_sample << " pow : " << pow(new_sample,1.7) << '\n';
        scaled_result = (2*COMPENSATION_MAX/PI) *std::atan(PI*INCLINATION*pow(new_sample,1.7)/(2*COMPENSATION_MAX));
    }
    else{
        scaled_result = (2*COMPENSATION_MAX/PI) *std::atan(PI*INCLINATION*new_sample/(2*COMPENSATION_MAX));
    }

    //scaled_result = (2*COMPENSATION_MAX/PI) *std::atan(PI*INCLINATION*new_sample/(2*COMPENSATION_MAX));
    // float threshold = 0.35f;
    // float ratio = 2.0f; // 압축 비율

    // if (new_sample < threshold) {
    //     scaled_result = (2*COMPENSATION_MAX/PI) *std::atan(PI*INCLINATION*new_sample/(2*COMPENSATION_MAX)); // 스케일링 없이 그대로 반환
    // } else {
    //     // 압축 적용
    //     float result = threshold + (new_sample - threshold) / ratio;
    //     scaled_result = (2*COMPENSATION_MAX/PI) *std::atan(PI*INCLINATION*result/(2*COMPENSATION_MAX));
    // }

    // float threshold = 0.5f;
    // float knee_width = 0.1f; // 무릎 폭
    // float value;
    // if (new_sample < (threshold - knee_width / 2)) {
    //     scaled_result = (2*COMPENSATION_MAX/PI) *std::atan(PI*INCLINATION*new_sample/(2*COMPENSATION_MAX)); // 스케일링 없이 그대로 반환
    // } else if (new_sample > (threshold + knee_width / 2)) {
    //     // 완전한 압축 적용
    //     float ratio = 2.0f;
    //     value = threshold + (new_sample - threshold) / ratio;
    //     scaled_result = (2*COMPENSATION_MAX/PI) *std::atan(PI*INCLINATION*value/(2*COMPENSATION_MAX));
    // } else {
    //     // 소프트 니 영역에서 압축 비율을 선형적으로 변경
    //     float ratio = 1.0f + ((new_sample - (threshold - knee_width / 2)) / knee_width) * (1.0f / 2.0f - 1.0f);
    //     value = threshold + (new_sample - threshold) / ratio;
    //     scaled_result = (2*COMPENSATION_MAX/PI) *std::atan(PI*INCLINATION*value/(2*COMPENSATION_MAX));
    // }

    return scaled_result;
}





std::vector<int32_t> RPY2DXL(double roll_f, double pitch_f, double yaw_f, double mouth_f, int mode)
{

  // CHANGE ROLL PITCH YAW MOUTH VALUES TO DXL POSITIONS
  //-------- change roll pitch yaw angle to string length L1 L2 L3

  double yaw_degree = yaw_f * 180 / PI;
  double roll = roll_f;
  double pitch = pitch_f;

  Eigen::MatrixXd R_X_roll(3, 3), R_Y_pitch(3, 3); // x축 rotation matrix, y축 rotation matrix
  R_X_roll  << 1,    0,          0,
               0,    cos(roll),  -sin(roll),
               0,    sin(roll),  cos(roll);

  R_Y_pitch << cos(pitch),   0,  sin(pitch),
               0,            1,  0,
               -sin(pitch),  0,  cos(pitch);

  Eigen::VectorXd zp(3), zn(3); // 바닥 평면 수직벡터, 머리뚜껑 평면 수직벡터

  zp << 0, 0, 1;
  zn = R_Y_pitch * R_X_roll * zp;
  double n1 = zn(0), n2 = zn(1);
  double theta = acos((zn.transpose() * zp).value()); // zp~zn 각도 (0 이상, 90 이하)
  double alpha = atan2(n2, n1); // x축~zn바닥projection 각도
  Eigen::VectorXd u_r(2); // zn바닥projection 방향
  u_r << cos(alpha), sin(alpha);

  if(theta <= 0.00001)
    theta = 0.001;
  
  double r = cfg_robot.height / theta; // 평면 중심 원격 회전 반경
  double r_x = r * cos(alpha); // 원격 회전 중심점 x,y
  double r_y = r * sin(alpha);

  Eigen::VectorXd R(2);

  R << r_x, r_y;

  Eigen::VectorXd P1(2), P2(2), P3(2);
  
  P1 << cfg_robot.hole_radius * cos(0), cfg_robot.hole_radius* sin(0); // 1번 구멍의 바닥위치
  P2 << cfg_robot.hole_radius * cos(2 * PI / 3), cfg_robot.hole_radius* sin(2 * PI / 3); // 2번 구멍의 바닥위치
  P3 << cfg_robot.hole_radius * cos(4 * PI / 3), cfg_robot.hole_radius* sin(4 * PI / 3); // 3번 구멍의 바닥위치

  Eigen::VectorXd RP1(2), RP2(2), RP3(2); // R -> Pi 벡터
  
  RP1 = P1 - R;
  RP2 = P2 - R;
  RP3 = P3 - R;

  double r1 = (-u_r.transpose() * RP1).value(); // Pi의 회전 반경
  double r2 = (-u_r.transpose() * RP2).value();
  double r3 = (-u_r.transpose() * RP3).value();
  double L1 = abs(r1) * theta; //앞쪽(DXL#1) // abs는 혹시 몰라서
  double L2 = abs(r2) * theta; //오른쪽(관찰자 시점//DXL#2)
  double L3 = abs(r3) * theta; //왼쪽(관찰자 시점//DXL#3)

  Eigen::VectorXd u_r3(3), p13(3), p23(3);

  u_r3 << cos(alpha), sin(alpha), 0;
  p13 << cfg_robot.hole_radius * cos(0), cfg_robot.hole_radius* sin(0), 0;
  p23 << cfg_robot.hole_radius * cos(2 * PI / 3), cfg_robot.hole_radius* sin(2 * PI / 3), 0;

  Eigen::VectorXd n0(3), n1p(3), n2p(3), n01p(3), n02p(3);

  n0 = r * (1 - cos(theta)) * u_r3 + r * sin(theta) * zp;
  n1p = p13 + r1 * (1 - cos(theta)) * u_r3 + r1 * sin(theta) * zp;
  n2p = p23 + r2 * (1 - cos(theta)) * u_r3 + r2 * sin(theta) * zp;
  n01p = n1p - n0;
  n02p = n2p - n0;

  Eigen::MatrixXd Rypr(3, 3), Pypr(3, 3), Nypr(3, 3);

  Pypr << p13(0), p23(0), zp(0),
          p13(1), p23(1), zp(1),
          p13(2), p23(2), zp(2);

  Nypr << n01p(0), n02p(0), zn(0),
          n01p(1), n02p(1), zn(1),
          n01p(2), n02p(2), zn(2);

  Rypr = Nypr * Pypr.inverse();

  double pitch_real = asin(-Rypr(2, 0));
  double yaw_real   = asin(Rypr(1, 0) / cos(pitch_real));
  
  //yaw compensation
  
  double yaw_real_degree = yaw_real / PI * 180;
  
  if (abs(yaw_real_degree) < 60)
    yaw_degree -= yaw_real_degree;
  else
    INFO_STREAM ( "Check kinematics yaw compensation value!" );

  //--------- string length to DXL position

  //---------

  double  dxl_goal_position_pitch_double = 0,
          dxl_goal_position_rollr_double = 0,
          dxl_goal_position_rolll_double = 0,
          dxl_goal_position_yaw_double   = 0, 
          dxl_goal_position_mouth_double = 0;

  double pitch_diff = (cfg_robot.height - L1) * (4096 / (cfg_robot.pulley_diameter * PI));

  dxl_goal_position_pitch_double = g_home.home_pitch - pitch_diff;

//double delta_mouth = (float)mouth_f * cfg_robot.mouth_tune; // mouth_f는 0 ~ 1 사이(보통 0.1 ~ 0.3), cfg_robot.mouth_tune은 입을 최대로 크게 벌리는 정도
  double delta_mouth = (float)mouth_f;

  dxl_goal_position_mouth_double = g_home.home_mouth - delta_mouth - pitch_diff / cfg_robot.mouth_pitch_compensation;

  if (mode == 0) // mirroring
  {
    dxl_goal_position_yaw_double = (-1) * static_cast<double> (yaw_degree) * cfg_robot.yaw_gear_ratio * 4096.0 / 360.0 + g_home.home_yaw; // 2
    // dxl_goal_position_yaw_double = static_cast<double> (yaw_degree) * 1 * 4096.0 / 360.0 + g_home.home_yaw; // 2

    double rollR_diff = (cfg_robot.height - L2) * (4096 / (cfg_robot.pulley_diameter * PI));
    double rollL_diff = (cfg_robot.height - L3) * (4096 / (cfg_robot.pulley_diameter * PI));
    if (pitch < 0)
    {
      pitch_diff *= 2;
      //rollR_diff *= 1.5;
      //rollL_diff *= 1.5;
    }
    //// pitch tension up (220805)
    // if (pitch < 0.2) 
    // {
    //   if (rollR_diff > 0)
    //     rollR_diff *= 1.3;
    //   if (rollL_diff > 0)
    //     rollL_diff *= 1.3;
    // }
    dxl_goal_position_rollr_double = g_home.home_roll_r - rollR_diff - (delta_mouth / cfg_robot.mouth_back_compensation); // 1.5
    dxl_goal_position_rolll_double = g_home.home_roll_l - rollL_diff - (delta_mouth / cfg_robot.mouth_back_compensation);

    //// R, L이 너무 기울어졌을 때 입 보상 끄는거
    //if (dxl_goal_position_rollr_double < 200) dxl_goal_position_rollr_double += delta_mouth / cfg_robot.mouth_back_compensation;
    //if (dxl_goal_position_rolll_double < 200) dxl_goal_position_rolll_double += delta_mouth / cfg_robot.mouth_back_compensation;
  }
  else if (mode == 1) // cloning
  {
    dxl_goal_position_yaw_double = static_cast<double> (yaw_degree) * cfg_robot.yaw_gear_ratio * 4096.0 / 360.0 + g_home.home_yaw; // 2
    double rollR_diff = (cfg_robot.height - L3) * (4096 / (cfg_robot.pulley_diameter * PI));
    double rollL_diff = (cfg_robot.height - L2) * (4096 / (cfg_robot.pulley_diameter * PI));
    // pitch tension up (220805)
    if (pitch < 0.2) 
    {
      if (rollR_diff > 0)
        rollR_diff *= 1.3;
      if (rollL_diff > 0)
        rollL_diff *= 1.3;
    }
    // R L 변화량 change
    dxl_goal_position_rollr_double = g_home.home_roll_r - rollR_diff - (delta_mouth / cfg_robot.mouth_back_compensation);
    dxl_goal_position_rolll_double = g_home.home_roll_l - rollL_diff - (delta_mouth / cfg_robot.mouth_back_compensation);
  }
  else 
  {
    INFO_STREAM( "RPY2DXL: CHECK MODE NUMBER" );
    dxl_goal_position_rollr_double = g_home.home_roll_r - (cfg_robot.height - L2) * (4096 / (cfg_robot.pulley_diameter * PI)) - (delta_mouth / cfg_robot.mouth_back_compensation); // 1.5
    dxl_goal_position_rolll_double = g_home.home_roll_l - (cfg_robot.height - L3) * (4096 / (cfg_robot.pulley_diameter * PI)) - (delta_mouth / cfg_robot.mouth_back_compensation);
  }
  
  std::vector<int32_t> DXL(5);
  DXL[0] = static_cast<int32_t>(std::lround(dxl_goal_position_pitch_double));
  DXL[1] = static_cast<int32_t>(std::lround(dxl_goal_position_rollr_double));
  DXL[2] = static_cast<int32_t>(std::lround(dxl_goal_position_rolll_double));
  DXL[3] = static_cast<int32_t>(std::lround(dxl_goal_position_yaw_double));
  DXL[4] = static_cast<int32_t>(std::lround(dxl_goal_position_mouth_double));
  
  return DXL;
}

int calculateDXLGoalVelocity_velocityBased(double current_position, double goal_position, double current_velocity, double profile_acceleration, double control_ms)
{
    // --- 단위 변환용 상수 (XC 계열 기준) ---
    // 이 상수는 특정 다이나믹셀 모델에 종속적임
    const double RPM_PER_VEL_UNIT = 0.229;
    const double RPM_SQ_PER_ACCEL_UNIT = 214.577;
    const double TICK_PER_REVOLUTION = 4096.0;
    const double MIN_PER_MS = 1.0 / 60000.0;

    // 계산용 단위: tick, minute
    const double TICK_PER_MIN_PER_VEL_UNIT = RPM_PER_VEL_UNIT * TICK_PER_REVOLUTION;
    const double TICK_PER_MIN_SQ_PER_ACCEL_UNIT = RPM_SQ_PER_ACCEL_UNIT * TICK_PER_REVOLUTION;
    const double VEL_UNIT_PER_TICK_PER_MIN = 1.0 / TICK_PER_MIN_PER_VEL_UNIT;

    // --- 입력 값을 계산용 단위로 변환 ---
    double a_mag = profile_acceleration * TICK_PER_MIN_SQ_PER_ACCEL_UNIT;
    double T = control_ms * MIN_PER_MS;
    double V_c = current_velocity * TICK_PER_MIN_PER_VEL_UNIT;
    double delta_p = goal_position - current_position;

    double goal_velocity;

    // --- 경계 조건 확인 ---
    // 제어 주기 동안 도달 가능한 변위의 최대/최소 범위 계산
    double delta_p_max = (V_c + a_mag * T / 2.0) * T;
    double delta_p_min = (V_c - a_mag * T / 2.0) * T;

    if (delta_p >= delta_p_max) {
        goal_velocity = V_c + a_mag * T; // 최대 속도
    } else if (delta_p <= delta_p_min) {
        goal_velocity = V_c - a_mag * T; // 최소 속도
    } else {
        // --- 목표가 도달 가능한 범위 내에 있는 경우 (이차방정식 풀이) ---

        double a; // 부호가 적용된 실제 가속도

        if (delta_p > V_c * T) { a = a_mag; } // 가속
        else if (delta_p < V_c * T) { a = -a_mag; } // 감속
        else {
            // 현재 속도로 목표 위치에 도달하는 경우
            goal_velocity = V_c;
        }

        if (delta_p != V_c * T) {
            // 이차방정식 계수 계산
            // Vg^2 - (2*a*T + 2*Vc)*Vg + (Vc^2 + 2*a*delta_p) = 0
            double A = 1.0;
            double B = -(2.0 * a * T + 2.0 * V_c);
            double C = V_c * V_c + 2.0 * a * delta_p;

            double discriminant = B * B - 4.0 * A * C;

            // 판별식 안전장치
            // (이론적으로는 위의 경계 처리로 인해 discriminant < 0이 될 수 없지만, 부동 소수점 연산 오류를 대비한 방어 코드)
            if (discriminant < 0) {
                // 경계 값에 근접한 경우, full 가/감속으로 처리
                if (a > 0) {
                    goal_velocity = V_c + a_mag * T;
                } else {
                    goal_velocity = V_c - a_mag * T;
                }
            }
            else {
                // 근의 공식 적용 및 해 선택
                double sqrt_discriminant = sqrt(discriminant);

                if (a > 0) {
                    // 가속 -> 마이너스 근 선택
                    goal_velocity = (-B - sqrt_discriminant) / (2.0 * A);
                } else {
                    // 감속 -> 플러스 근 선택
                    goal_velocity = (-B + sqrt_discriminant) / (2.0 * A);
                }
            }
        }
    }

    // --- 계산된 속도를 DXL 단위로 변환 후 반환 ---
    return static_cast<int>(round(goal_velocity * VEL_UNIT_PER_TICK_PER_MIN));
}

int calculateDXLGoalVelocity_timeBased_ds(double current_position, double goal_position, double current_velocity, double profile_acceleration, double control_ms)
{
    // --- 단위 변환용 상수 (XC 계열 기준) ---
    const double RPM_PER_VEL_UNIT = 0.229;
    const double TICK_PER_REVOLUTION = 4096.0;
    const double MIN_PER_MS = 1.0 / 60000.0;

    // 계산용 단위: tick, minute
    const double TICK_PER_MIN_PER_VEL_UNIT = RPM_PER_VEL_UNIT * TICK_PER_REVOLUTION;
    const double VEL_UNIT_PER_TICK_PER_MIN = 1.0 / TICK_PER_MIN_PER_VEL_UNIT;

    // --- 입력 값 유효성 검사 및 조정 ---
    // 총 제어 시간이 0 이하면 움직일 수 없으므로 현재 속도 반환
    if (control_ms <= 0) {
        std::cerr << "Warning: Control time is zero or negative. Returning current velocity." << std::endl;
        return static_cast<int>(std::round(current_velocity));
    }
    // 가속 시간은 총 제어 시간을 넘을 수 없음
    if (profile_acceleration > control_ms) {
        std::cerr << "Warning: Profile acceleration time exceeds control time. Adjusting to control time." << std::endl;
        profile_acceleration = control_ms;
    }

    // --- 입력 값을 계산용 단위로 변환 ---
    double s = goal_position - current_position; // 총 이동 거리 (tick)
    double Vc = current_velocity * TICK_PER_MIN_PER_VEL_UNIT; // 현재 속도 (tick/min)
    double Ta = profile_acceleration * MIN_PER_MS; // 가속 시간 (min)
    double T_total = control_ms * MIN_PER_MS; // 총 이동 시간 (min)

    double Vg; // 계산할 목표 속도 (tick/min)

    // --- 핵심 로직: 목표 속도 계산 ---
    // 공식: Vg = (2*s - Vc*Ta) / (2*T_total - Ta)
    double numerator = 2.0 * s - Vc * Ta;
    double denominator = 2.0 * T_total - Ta;

    // 분모가 0이 되는 경우 방지 (부동 소수점 비교)
    if (std::abs(denominator) < 1e-9) {
        // 이 경우는 Ta가 T_total의 2배일 때 발생하며, 물리적으로 불가능한 상황.
        // 안전하게 등가속 운동으로 간주하고 계산
        std::cerr << "Warning: Denominator in velocity calculation is zero. Using alternative calculation." << std::endl;
        if (std::abs(T_total) < 1e-9) { // 총 시간이 0에 가까우면
             Vg = Vc; // 속도 변화 없음
        } else {
             // 등가속 공식: s = (Vc + Vg)/2 * T_total  => Vg = (2*s / T_total) - Vc
             Vg = (2.0 * s / T_total) - Vc;
        }
    } else {
        Vg = numerator / denominator;
    }

    // --- 계산된 속도를 DXL 단위로 변환 후 반환 ---
    return static_cast<int>(std::round(Vg * VEL_UNIT_PER_TICK_PER_MIN));
}

int calculateDXLGoalVelocity_timeBased_ff(double current_position_real, double current_position_desired, double goal_position, double control_ms, double Kp)
{
    // --- 단위 변환용 상수 (XC 계열 기준) ---
    const double RPM_PER_VEL_UNIT = 0.229;
    const double TICK_PER_REVOLUTION = 4096.0;
    const double MIN_PER_MS = 1.0 / 60000.0;

    // 계산용 단위: tick, minute
    const double TICK_PER_MIN_PER_VEL_UNIT = RPM_PER_VEL_UNIT * TICK_PER_REVOLUTION;
    const double VEL_UNIT_PER_TICK_PER_MIN = 1.0 / TICK_PER_MIN_PER_VEL_UNIT;

    // --- 입력 값을 계산용 단위로 변환 ---
    double T_total = control_ms * MIN_PER_MS; // 총 이동 시간 (min)

    // 피드포워드 속도 (tick/min)
    double V_ff = (goal_position - current_position_desired) / T_total;

    // 피드백 속도 (tick/min)
    double position_error = current_position_desired - current_position_real;
    double V_fb = position_error * Kp;
    // 전체 목표 속도 (tick/min)
    double Vg = V_ff + V_fb;

    // --- 계산된 속도를 DXL 단위로 변환 후 반환 ---
    return static_cast<int>(std::round(Vg * VEL_UNIT_PER_TICK_PER_MIN));
}

// DXL_goal_position update
void update_DXL_goal_position(int DXL_goal_position[], int DXL_1, int DXL_2, int DXL_3, int DXL_4, int DXL_5)
{
    DXL_goal_position[0] = DXL_1;
    DXL_goal_position[1] = DXL_2;
    DXL_goal_position[2] = DXL_3;
    DXL_goal_position[3] = DXL_4;
    DXL_goal_position[4] = DXL_5;
}

// assignClassWith1DMiddleBoundary 함수
int assignClassWith1DMiddleBoundary(double x, const vector<double>& boundaries) {
    for (size_t i = 0; i < boundaries.size(); ++i) {
        if (x < boundaries[i]) {
            return i; // 해당 경계값에 해당하는 클래스 반환
        }
    }
    return boundaries.size(); // 마지막 클래스
}
double calculateRMS(const vector<float>& data, size_t start, size_t frame_length) {
    double sum_of_squares = 0.0;

    for (size_t i = start; i < start + frame_length; ++i) {
        sum_of_squares += data[i] * data[i];
    }

    return sqrt(sum_of_squares / frame_length);
}
// getSegmentAverageGrad 함수
double getSegmentAverageGrad(const vector<float>& data, const string& delta, const string& mode) {
    vector<double> grad; // 기울기를 저장할 벡터

    if (delta == "one2one") {
        // one2one: 연속된 데이터 간의 차이 계산
        for (size_t i = 1; i < data.size(); ++i) {
            grad.push_back(data[i] - data[i - 1]);
        }
    } else if (delta == "end2end") {
        // end2end: 마지막 값과 첫 번째 값의 차이 계산
        if (!data.empty()) {
            grad.push_back(data.back() - data.front());
        }
    } else {
        cerr << "getSegmentAverageGrad delta error" << endl;
        return 0;
    }

    // mode에 따라 처리
    if (mode == "abs") {
        // 절대값 처리
        for (double& g : grad) {
            g = fabs(g);
        }
    } else if (mode == "pos") {
        // 양수만 남기기
        vector<double> positive_grad;
        for (double g : grad) {
            if (g > 0) {
                positive_grad.push_back(g);
            }
        }
        grad = positive_grad;
    } else if (mode == "neg") {
        // 음수만 남기기
        vector<double> negative_grad;
        for (double g : grad) {
            if (g < 0) {
                negative_grad.push_back(g);
            }
        }
        grad = negative_grad;
    } else if (mode == "org") {
        // 원본 그대로 사용
    } else {
        cerr << "getSegmentAverageGrad mode error" << endl;
        return 0;
    }

    // 기울기 벡터가 비어있는 경우
    if (grad.empty()) {
        return 0;
    }

    // 기울기의 평균값 반환
    double sum = accumulate(grad.begin(), grad.end(), 0.0);
    return sum / grad.size();
}

// getNextSegment_SegSeg 함수
vector<vector<double>> getNextSegment_SegSeg(
    const vector<double>& PrevEndOneBefore,
    const vector<double>& PrevEnd,
    const cnpy::NpyArray& segment,
    bool gradient,
    bool gotoZero
    ) {
    vector<size_t> shape = segment.shape;
    size_t K = shape[0]; // 타임스텝 (9)
    size_t D = shape[1]; // Roll, Pitch, Yaw (3)
    size_t N = shape[2]; // 슬라이스 개수

    const double* segmentData = segment.data<double>();

    size_t distSelectNum = 20;
    double distSelectDist = 0.22;
    size_t gradSelectNum = 15;
    size_t randomChooseNum = 10;

    distSelectNum = min(distSelectNum, N);
    
    // 거리 기반 후보 선택
    vector<double> distances(N, 0.0);
    vector<size_t> distIndices;

    // 거리 계산
    for (size_t n = 0; n < N; ++n) {
        double dist = 0.0;
        for (size_t d = 0; d < D; ++d) {
            double diff = segmentData[0 * D * N + d * N + n] - PrevEnd[d]; // - gradFinal[d];
            dist += diff * diff;
        }
        distances[n] = sqrt(dist);
        if (distances[n] < distSelectDist) {
            distIndices.push_back(n);
        }
    }

    // 거리 조건을 만족하는 후보가 randomChooseNum보다 적으면 가장 가까운 세그먼트를 선택
    if (distIndices.size() < randomChooseNum) {
        cout << "최소 거리 기준을 충족하는 세그먼트가 충분하지 않습니다. 가장 가까운 세그먼트를 선택합니다." << endl;
        cout << "Number of valid indices: " << distIndices.size() << endl;
        auto minIt = min_element(distances.begin(), distances.end());
        size_t minIndex = distance(distances.begin(), minIt);

        vector<vector<double>> selectedSegment(K, vector<double>(D, 0.0));
        for (size_t k = 0; k < K; ++k) {
            for (size_t d = 0; d < D; ++d) {
                selectedSegment[k][d] = segmentData[k * D * N + d * N + minIndex];
            }
        }
        return selectedSegment;
    }

    // 거리 기준으로 후보들 정렬
    sort(distIndices.begin(), distIndices.end(), [&distances](size_t a, size_t b) {
        return distances[a] < distances[b];
    });

    // 상위 distSelectNum 개수까지만 남김
    if (distIndices.size() > distSelectNum) {
        distIndices.resize(distSelectNum);
    }

    vector<size_t> finalIndices = distIndices;

    // Gradient 기반 후보 선택
    if (gradient) {
        gradSelectNum = std::min(gradSelectNum, distIndices.size());

        // 이전 세그먼트의 마지막 그래디언트 계산
        vector<double> gradFinal(D, 0.0);
        for (size_t d = 0; d < D; ++d) {
            gradFinal[d] = PrevEnd[d] - PrevEndOneBefore[d];
        }

        // 각 후보 세그먼트의 시작 그래디언트와의 거리 계산
        vector<pair<double, size_t>> gradDists; // {그래디언트 거리, 원본 인덱스}
        for (size_t idx : distIndices) {
            vector<double> startGrad(D);
            for (size_t d = 0; d < D; ++d) {
                double p1 = segmentData[0 * D * N + d * N + idx];
                double p2 = segmentData[1 * D * N + d * N + idx];
                startGrad[d] = p2 - p1;
            }
            // 그래디언트 거리 계산
            double dist_sq = 0.0;
            for (size_t d = 0; d < D; ++d) {
                double diff = startGrad[d] - gradFinal[d];
                dist_sq += diff * diff;
            }
            gradDists.push_back({sqrt(dist_sq), idx});
        }

        // 그래디언트 거리 기준으로 정렬
        sort(gradDists.begin(), gradDists.end());

        // 정렬된 인덱스를 gradIndices에 저장
        vector<size_t> gradIndices;
        for (const auto& pair : gradDists) {
            gradIndices.push_back(pair.second);
        }

        // 상위 gradSelectNum 개수까지만 남김
        if (gradIndices.size() > gradSelectNum) {
            gradIndices.resize(gradSelectNum);
        }

        finalIndices = gradIndices;

        if (gotoZero) {
            vector<pair<double, size_t>> gotoZeroScores; // {점수, 원본 인덱스}
            for (size_t idx : gradIndices) {
                double score = 0.0;
                for (size_t d = 0; d < D; ++d) {
                    double startPoint = segmentData[0 * D * N + d * N + idx];
                    double endPoint = segmentData[(K - 1) * D * N + d * N + idx];
                    score += startPoint * (endPoint - startPoint);
                }
                gotoZeroScores.push_back({score, idx});
            }

            // gotoZero 점수 기준으로 정렬
            sort(gotoZeroScores.begin(), gotoZeroScores.end());

            // 정렬된 인덱스를 finalIndices에 저장
            finalIndices.clear();
            for (const auto& pair : gotoZeroScores) {
                finalIndices.push_back(pair.second);
            }
        }
    }

    // 최종 세그먼트 선택
    randomChooseNum = std::min(randomChooseNum, finalIndices.size());

    if (randomChooseNum == 0) { // 후보가 없는 경우
        // 이 경우 distIndices에서 가장 좋은 것(첫번째)을 선택
        size_t chosenIndex = distIndices[0];

        vector<vector<double>> selectedSegment(K, vector<double>(D));
        for (size_t k = 0; k < K; ++k) {
            for (size_t d = 0; d < D; ++d) {
                selectedSegment[k][d] = segmentData[k * D * N + d * N + chosenIndex];
            }
        }
        return selectedSegment;
    }

    // 최종 후보들 중 랜덤으로 하나 선택
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, randomChooseNum - 1);
    size_t chosenIndex = finalIndices[dis(gen)];
    //cout << "choseIndex : " << chosenIndex <<'\n';
    //cout << "Chosen index: " << chosenIndex << endl;
    if (chosenIndex >= N) {
        cerr << "Error: Chosen index out of range. Check random selection logic." << endl;
        return vector<vector<double>>(); // 빈 결과 반환
    }

    vector<vector<double>> selectedSegment(K, vector<double>(D, 0.0));
    for (size_t k = 0; k < K; ++k) {
        for (size_t d = 0; d < D; ++d) {
            selectedSegment[k][d] = segmentData[k * D * N + d * N + chosenIndex];
        }
    }

    return selectedSegment;
}

vector<vector<double>> multExpToSegment(
    const vector<float>& ex_energy,
    vector<vector<double>> ex_segment,
    float threshold,
    float div
) {
    vector<int> decidenovocide(ex_segment.size(), 0);
    vector<int> novoicesteps(ex_segment.size(), 0);
    vector<float> novoiceexp(ex_segment.size(), 0.0);

    // 1. 음성 여부 판단
    for (size_t i = 0; i < ex_energy.size(); ++i) {
        decidenovocide[i] = (ex_energy[i] >= threshold) ? 1 : 0;
    }

    // 2. 무성 구간 스텝 계산
    int novoice = 0;
    for (size_t i = 0; i < decidenovocide.size(); ++i) {
        if (decidenovocide[i] == 0) {
            ++novoice;
        } else {
            novoice = 0;
        }
        novoicesteps[i] = novoice;
    }

    // 3. 무성 구간 지수 감쇠
    for (size_t i = 0; i < novoicesteps.size(); ++i) {
        novoiceexp[i] = exp(-static_cast<float>(novoicesteps[i]) / div);
        //cout << "novoiceexp[" << i << "] : " << novoiceexp[i] << '\n';
        if (novoiceexp[i] < 0.3) {
            novoiceexp[i] = 0.3;
        }
    }

    // 4. 세그먼트 업데이트
    for (size_t i = 1; i < ex_segment.size(); ++i) {
        for (size_t d = 0; d < ex_segment[i].size(); ++d) {
            double delta = ex_segment[i][d] - ex_segment[i - 1][d];
            ex_segment[i][d] = ex_segment[i - 1][d] + novoiceexp[i] * delta;

           //cout << "ex_segment[" << i << "] : " << ex_segment[i][d] << " delta : " << delta << '\n';
        }
        
    }

    return ex_segment;
}

// Helper function to convert std::vector to Eigen::VectorXd
VectorXd toEigenVector(const vector<double>& stdVec) {
    VectorXd eigenVec(stdVec.size());
    for (size_t i = 0; i < stdVec.size(); ++i) {
        eigenVec[i] = stdVec[i];
    }
    return eigenVec;
}

vector<vector<double>> connectTwoSegments(
    const vector<vector<double>>& prevSegment,
    const vector<vector<double>>& nextSegment,
    int n_new,
    int n_anchor_past,
    int n_anchor_future
) {
    // --- 안정성 및 입력 값 검증 ---
    if (n_anchor_past + n_anchor_future < 4) {
        cerr << "Error: Total anchors must be at least 4 for cubic spline interpolation." << endl;
        return nextSegment;
    }
    if (prevSegment.size() < n_anchor_past) {
        cerr << "Error: prevSegment does not have enough points for n_anchor_past." << endl;
        return nextSegment;
    }
    if (nextSegment.size() < n_new + n_anchor_future) {
        cerr << "Error: nextSegment does not have enough points for n_new and n_anchor_future." << endl;
        return nextSegment;
    }

    size_t D = nextSegment[0].size(); // 차원 (Roll, Pitch, Yaw)
    vector<vector<double>> interpolatedSegment = nextSegment; // 수정할 세그먼트 복사본 생성

    for (size_t d = 0; d < D; ++d) {
        vector<double> x_interpolate, y_interpolate;

        // prevSegment의 끝점과 nextSegment의 시작점이 일치하도록 nextSegment 전체를 평행 이동
        // double last_val_prev = prevSegment.back()[d];
        // double first_val_next = nextSegment[0][d];
        // double interval = last_val_prev - first_val_next;

        // for (size_t i = 0; i < nextSegment.size(); ++i) {
        //     interpolatedSegment[i][d] += interval;
        // }

        // --- 앵커 포인트(제어점) 설정 ---
        // 과거 앵커: prevSegment의 끝에서 n_anchor_past개의 포인트를 가져옴
        for (int i = 0; i < n_anchor_past; ++i) {
            x_interpolate.push_back(i);
            y_interpolate.push_back(prevSegment[prevSegment.size() - n_anchor_past + i][d]);
        }

        // 미래 앵커: nextSegment에서 새로 생성될 n_new개의 포인트 이후 지점부터 n_anchor_future개를 가져옴
        for (int i = 0; i < n_anchor_future; ++i) {
            // x 좌표는 보간 구간 뒤에 위치해야 함
            x_interpolate.push_back(n_anchor_past + n_new + i);
            // y 값은 n_new 인덱스부터 가져옴
            y_interpolate.push_back(interpolatedSegment[n_new + i][d]);
        }
        
        // Eigen Vector로 변환
        VectorXd x = toEigenVector(x_interpolate);
        VectorXd y = toEigenVector(y_interpolate);

        // --- 3차 스플라인 피팅 ---
        int degree = 3;
        Eigen::Spline<double, 1> spline = Eigen::SplineFitting<Eigen::Spline<double, 1>>::Interpolate(y.transpose(), degree, x);

        // --- 새로운 포인트 보간 및 대체 ---
        // nextSegment의 첫 n_new 개의 포인트를 새로운 값으로 대체
        for (int i = 0; i < n_new; ++i) {
            // 보간할 지점의 x값(t)은 과거 앵커와 미래 앵커 사이에 위치
            double t = static_cast<double>(n_anchor_past + i);

            Eigen::Spline<double, 1>::PointType result = spline(t);
            interpolatedSegment[i][d] = result(0);
        }
    }

    return interpolatedSegment;
}

// AM_fun 함수
float AM_fun(float min_open, float B, float r_k, float r_k_1, float r_k_2, float lim_delta_r) {
    float delta_r = r_k - r_k_1;
    float B_adt;

    // B_adt 계산
    if (fabs(delta_r) < lim_delta_r) {
        B_adt = B * (2 - fabs(delta_r / lim_delta_r));
    } else {
        B_adt = 1 * B;
    }

    float c = 0;
    if ((r_k - r_k_1) * (r_k_1 - r_k_2) <= 0) {
        return c = r_k;
    }

    // 교점 계산
    float A_adt = 0;
    for (float A = 0; A <= 1; A += 0.01) { // A 값은 0부터 1까지 0.01씩 증가
        if (A * tanh(B_adt * (min_open - 0.01)) > min_open - 0.01 && A * tanh(B_adt * (min_open + 0.01)) < min_open + 0.01) {
            A_adt = A;
            break;
        }
    }

    // 증가/감소에 따른 처리
    if (r_k > r_k_1 && r_k_1 >= r_k_2) {
        c = A_adt * tanh(B_adt * r_k);
    }

    if (r_k < r_k_1 && r_k_1 <= r_k_2) {
        c = (1 / B_adt) * atanh(r_k / A_adt);
    }

    return c;
}

std::tuple<float, float, float> lin_fit_fun2(float S, float X_pre, float grad_up_pre, float grad_down_pre, float del_grad, float dt) {
    float X_now = 0.0f;
    float grad_up_now = 0.0f;
    float grad_down_now = 0.0f;

    if (S > X_pre) {
        // 증가하는 경우
        grad_up_now = del_grad * (S - X_pre);
        grad_down_now = 0.0f;
        X_now = X_pre + grad_up_now * dt;
    } else {
        // 감소하는 경우
        grad_down_now = del_grad * (S - X_pre);
        grad_up_now = 0.0f;
        X_now = X_pre + grad_down_now * dt;
    }

    // X_now가 0보다 작으면 0으로 설정
    if (X_now < 0.0f) {
        X_now = 0.0f;
        grad_up_now = 0.0f;
        grad_down_now = 0.0f;
    }

    return std::make_tuple(X_now, grad_up_now, grad_down_now);
}