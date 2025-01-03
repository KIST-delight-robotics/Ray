#include "Macro_function.h"


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

float calculate_mouth(float up2mouth, float max_MOUTH, float min_MOUTH) {

    float mouth = max_MOUTH * up2mouth - up2mouth * (max_MOUTH - min_MOUTH)/0.7;
    return mouth;
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





std::vector<int> RPY2DXL(double roll_f, double pitch_f, double yaw_f, double mouth_f, int mode)
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
  
  double r = ROBOT_HEIGHT / theta; // 평면 중심 원격 회전 반경
  double r_x = r * cos(alpha); // 원격 회전 중심점 x,y
  double r_y = r * sin(alpha);

  Eigen::VectorXd R(2);

  R << r_x, r_y;

  Eigen::VectorXd P1(2), P2(2), P3(2);
  
  P1 << ROBOT_HOLE_RADIUS * cos(0), ROBOT_HOLE_RADIUS* sin(0); // 1번 구멍의 바닥위치
  P2 << ROBOT_HOLE_RADIUS * cos(2 * PI / 3), ROBOT_HOLE_RADIUS* sin(2 * PI / 3); // 2번 구멍의 바닥위치
  P3 << ROBOT_HOLE_RADIUS * cos(4 * PI / 3), ROBOT_HOLE_RADIUS* sin(4 * PI / 3); // 3번 구멍의 바닥위치

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
  p13 << ROBOT_HOLE_RADIUS * cos(0), ROBOT_HOLE_RADIUS* sin(0), 0;
  p23 << ROBOT_HOLE_RADIUS * cos(2 * PI / 3), ROBOT_HOLE_RADIUS* sin(2 * PI / 3), 0;

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

  double pitch_diff = (ROBOT_HEIGHT - L1) * (4096 / (PULLY_DIAMETER * PI));

  dxl_goal_position_pitch_double = DEFAULT_PITCH - pitch_diff;

  double delta_mouth = (float)mouth_f * ROBOT_MOUTH_TUNE; // 250

  dxl_goal_position_mouth_double = DEFAULT_MOUTH - delta_mouth - pitch_diff / ROBOT_MOUTH_PITCH_COMPENSATION;

  if (mode == 0) // mirroring
  {
    //dxl_goal_position_yaw_double = (-1) * static_cast<double> (yaw_degree) * ROBOT_YAW_GEAR_RATIO * 4096.0 / 360.0 + default_YAW; // 2
    dxl_goal_position_yaw_double = static_cast<double> (yaw_degree) * 1 * 4096.0 / 360.0 + DEFAULT_YAW; // 2

    double rollR_diff = (ROBOT_HEIGHT - L2) * (4096 / (PULLY_DIAMETER * PI));
    double rollL_diff = (ROBOT_HEIGHT - L3) * (4096 / (PULLY_DIAMETER * PI));
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
    dxl_goal_position_rollr_double = DEFAULT_ROLL_R - rollR_diff - (delta_mouth / ROBOT_MOUTH_BACK_COMPENSATION); // 1.5
    dxl_goal_position_rolll_double = DEFAULT_ROLL_L - rollL_diff - (delta_mouth / ROBOT_MOUTH_BACK_COMPENSATION);

    //// R, L이 너무 기울어졌을 때 입 보상 끄는거
    //if (dxl_goal_position_rollr_double < 200) dxl_goal_position_rollr_double += delta_mouth / ROBOT_MOUTH_BACK_COMPENSATION;
    //if (dxl_goal_position_rolll_double < 200) dxl_goal_position_rolll_double += delta_mouth / ROBOT_MOUTH_BACK_COMPENSATION;
  }
  else if (mode == 1) // cloning
  {
    dxl_goal_position_yaw_double = static_cast<double> (yaw_degree) * ROBOT_YAW_GEAR_RATIO * 4096.0 / 360.0 + DEFAULT_YAW; // 2
    double rollR_diff = (ROBOT_HEIGHT - L3) * (4096 / (PULLY_DIAMETER * PI));
    double rollL_diff = (ROBOT_HEIGHT - L2) * (4096 / (PULLY_DIAMETER * PI));
    // pitch tension up (220805)
    if (pitch < 0.2) 
    {
      if (rollR_diff > 0)
        rollR_diff *= 1.3;
      if (rollL_diff > 0)
        rollL_diff *= 1.3;
    }
    // R L 변화량 change
    dxl_goal_position_rollr_double = DEFAULT_ROLL_R - rollR_diff - (delta_mouth / ROBOT_MOUTH_BACK_COMPENSATION);
    dxl_goal_position_rolll_double = DEFAULT_ROLL_L - rollL_diff - (delta_mouth / ROBOT_MOUTH_BACK_COMPENSATION);
  }
  else 
  {
    INFO_STREAM( "RPY2DXL: CHECK MODE NUMBER" );
    dxl_goal_position_rollr_double = DEFAULT_ROLL_R - (ROBOT_HEIGHT - L2) * (4096 / (PULLY_DIAMETER * PI)) - (delta_mouth / ROBOT_MOUTH_BACK_COMPENSATION); // 1.5
    dxl_goal_position_rolll_double = DEFAULT_ROLL_L - (ROBOT_HEIGHT - L3) * (4096 / (PULLY_DIAMETER * PI)) - (delta_mouth / ROBOT_MOUTH_BACK_COMPENSATION);
  }
  
  std::vector<int> DXL(5);
  DXL[0] = (int)dxl_goal_position_pitch_double;
  DXL[1] = (int)dxl_goal_position_rollr_double;
  DXL[2] = (int)dxl_goal_position_rolll_double;
  DXL[3] = (int)dxl_goal_position_yaw_double;
  DXL[4] = (int)dxl_goal_position_mouth_double;
  
  return DXL;

}

// Move 5 DXL to DXL_goal_position with specified velocity and acceleration
bool moveDXLtoDesiredPosition(
    dynamixel::GroupSyncWrite& groupSyncWriteVelocity,
    dynamixel::GroupSyncWrite& groupSyncWritePosition,
    int DXL_ID[],
    int goal_position[],
    int velocity)
{
    uint8_t param_profile_velocity[4];
    //groupSyncWriteVelocity.clearParam();
    // 속도와 가속도 값을 4바이트 배열로 변환
    trans_int2bin_4(param_profile_velocity, velocity);

    
     
    // 각 모터에 대해 프로파일 속도 값을 SyncWrite에 추가
    for (int i = 0; i < DXL_NUM; i++)
    {
        if (!groupSyncWriteVelocity.addParam(DXL_ID[i], param_profile_velocity))
        {
            fprintf(stderr, "[ID:%03d] groupSyncWriteVelocity addParam failed\n", DXL_ID[i]);
            return false;
        }
    }

    int comm_result = groupSyncWriteVelocity.txPacket();
    // 프로파일 속도 값 전송
    if (comm_result != COMM_SUCCESS)
    {
        fprintf(stderr, "Failed to send profile velocity values. Error code: %d\n", comm_result);
        return false;
    }
    groupSyncWriteVelocity.clearParam();
    //groupSyncWritePosition.clearParam();

    // 각 모터에 대해 목표 위치 값을 SyncWrite에 추가
    for (int i = 0; i < DXL_NUM; i++)
    {
        uint8_t param_goal_position[4];
        trans_int2bin_4(param_goal_position, goal_position[i]);

        if (!groupSyncWritePosition.addParam(DXL_ID[i], param_goal_position))
        {
            fprintf(stderr, "[ID:%03d] groupSyncWritePosition addParam failed\n", DXL_ID[i]);
            return false;
        }
    }

    // 목표 위치 값 전송
    if (groupSyncWritePosition.txPacket() != COMM_SUCCESS)
    {
        fprintf(stderr, "Failed to send goal position values\n");
        return false;
    }
    groupSyncWritePosition.clearParam();
    //std::this_thread::sleep_for(std::chrono::milliseconds(9));
    return true;
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

//int를 32bit bin로 변경 (little endian)
void trans_int2bin_4(uint8_t param_goal_position[4], int dxl_goal_position)
{
    param_goal_position[0] = DXL_LOBYTE(DXL_LOWORD(dxl_goal_position));
    param_goal_position[1] = DXL_HIBYTE(DXL_LOWORD(dxl_goal_position));
    param_goal_position[2] = DXL_LOBYTE(DXL_HIWORD(dxl_goal_position));
    param_goal_position[3] = DXL_HIBYTE(DXL_HIWORD(dxl_goal_position));
}

int enable_torque(dynamixel::PacketHandler *packetHandler, dynamixel::PortHandler *portHandler, int *DXL_ID, uint8_t dxl_error)
{
    int dxl_comm_result = COMM_TX_FAIL;

  for(int i = 0; i < DXL_NUM; i++)
  {
    dxl_comm_result = packetHandler->write1ByteTxRx(portHandler, DXL_ID[i], ADDR_PRO_TORQUE_ENABLE, TORQUE_ENABLE, &dxl_error);
    if (dxl_comm_result != COMM_SUCCESS)
    {
      printf("%s\n", packetHandler->getTxRxResult(dxl_comm_result));
    }
    else if (dxl_error != 0)
    {
      printf("%s\n", packetHandler->getRxPacketError(dxl_error));
    }
    else
    {
      printf("Dynamixel#%d -----> torque ON \n", DXL_ID[i]);
    }
  }
  return 1;
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
    //size_t gradSelectNum = 15;
    size_t randomChooseNum = 10;
    randomChooseNum = min(randomChooseNum, distSelectNum);

    vector<double> distances(N, 0.0);
    vector<size_t> indices;
    vector<double> gradFinal(D, 0.0);
    for (size_t d = 0; d < D; ++d) {
        gradFinal[d] = PrevEnd[d] - PrevEndOneBefore[d];
    }

    // 거리 계산
    for (size_t n = 0; n < N; ++n) {
        double dist = 0.0;
        for (size_t d = 0; d < D; ++d) {
            double diff = segmentData[0 * D * N + d * N + n] - PrevEnd[d]; // - gradFinal[d];
            dist += diff * diff;
        }
        distances[n] = sqrt(dist);
        if (distances[n] < distSelectDist) {
            indices.push_back(n);
        }
    }
    
    cout << "Number of valid indices: " << indices.size() << endl;
    if (indices.empty()) {
        cerr << "Error: No valid indices found. Check distance threshold or input data." << endl;
        return vector<vector<double>>(); // 빈 결과 반환
    }

    sort(indices.begin(), indices.end(), [&distances](size_t a, size_t b) {
        return distances[a] < distances[b];
    });
    randomChooseNum = std::min(randomChooseNum, indices.size());

    // 랜덤 인덱스 선택
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, randomChooseNum - 1);
    size_t chosenIndex = indices[dis(gen)];
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
    vector<int> decidenovocide(ex_energy.size(), 0);
    vector<int> novoicesteps(ex_energy.size(), 0);
    vector<float> novoiceexp(ex_energy.size(), 0.0);

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
    const vector<double>& PrevEndOneBefore,
    const std::array<double, 3>& lastValues,
    const vector<vector<double>>& nextSegment,
    int n_interpolate
) {
    size_t D = nextSegment[0].size(); // Roll, Pitch, Yaw dimensions
    int n_new = n_interpolate; // n_new를 n_interpolate와 동일하게 설정
    double interval[3];
    size_t totalSize = n_new + (nextSegment.size() - n_new); 
    vector<vector<double>> interpolatedSegment(totalSize, vector<double>(D, 0.0));


    for (size_t d = 0; d < D; ++d) {
        vector<double> x_interpolate, y_interpolate;

        interval[d] = lastValues[d]-nextSegment[0][d];
        //cout << "interval : " << interval[d] << '\n';

        x_interpolate.push_back(0);
        x_interpolate.push_back(1);

        for (size_t i = 0; i < nextSegment.size(); ++i) {
            interpolatedSegment[i][d] = nextSegment[i][d]+interval[d];
        }

         y_interpolate.push_back(PrevEndOneBefore[d]);
         y_interpolate.push_back(lastValues[d]);

        for(int i = 3; i<=5; i++){
            x_interpolate.push_back(i+1);
            double nextValue = interpolatedSegment[i - 1][d];
            if (std::isnan(nextValue)) {
                cerr << "Warning: nextSegment[" << i - 1 << "][" << d << "] is nan, replacing with 0.0" << endl;
                nextValue = 0.0;
            }
            y_interpolate.push_back(nextValue);
        }

        // Eigen Vector로 변환
        VectorXd x = toEigenVector(x_interpolate);
        VectorXd y = toEigenVector(y_interpolate);
        //cout << "x_interpolate: ";
        //for (double val : x_interpolate) cout << val << " ";
        //cout << endl;

        // cout << "y_interpolate: ";
        // for (double val : y_interpolate) cout << val << " ";
        // cout << endl;

        // 스플라인 피팅
        int degree = 3;
        Eigen::RowVectorXd y_row = y.transpose();
        Eigen::Spline<double, 1> spline = Eigen::SplineFitting<Eigen::Spline<double, 1>>::Interpolate(y_row, degree, x);

        for(int i = 1; i<= 2; i++){
            double t = static_cast<double>(i+1);
            if (t < x.minCoeff() || t > x.maxCoeff()) {
                cerr << "Error: t is out of range. t: " << t 
                    << ", range: [" << x.minCoeff() << ", " << x.maxCoeff() << "]" << endl;
                continue;
            }

            Eigen::Spline<double, 1>::PointType result = spline(t);
            double interpolatedValue = result(0);

            interpolatedSegment[i - 1][d] = interpolatedValue;

            // 결과 출력
            //cout << "InterpolatedSegment[" << i - 1 << "][" << d << "] = " << interpolatedValue << endl;
            
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
        del_grad = 15.0f * (S - X_pre);
        grad_up_now = del_grad;
        grad_down_now = 0.0f;
        X_now = X_pre + grad_up_now * dt;
    } else {
        // 감소하는 경우
        del_grad = 15.0f * (S - X_pre);
        grad_down_now = del_grad;
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