#ifndef DYNAMIXEL_DRIVER_H 
#define DYNAMIXEL_DRIVER_H

#include "dynamixel_sdk.h"
#include <mutex>    // std::mutex
#include <vector>   // std::vector
#include <string>   // std::string
#include <cstdint>  // std::uint8_t, std::int16_t, std::int32_t

// 로깅 및 상태 모니터링을 위한 구조체
struct MotorState {
    int32_t position;
    int32_t velocity;
    int16_t current; // Load
};

class DynamixelDriver {
private:
    dynamixel::PortHandler* portHandler_;
    dynamixel::PacketHandler* packetHandler_;
    
    std::vector<uint8_t> dxl_ids_;
    std::string device_name_;
    float protocol_version_;

    // 통신 충돌 방지용 뮤텍스
    std::mutex dxl_mutex_;

    // GroupHandler 객체들
    dynamixel::GroupSyncWrite *groupSyncWritePos_;
    dynamixel::GroupSyncWrite *groupSyncWriteVel_;
    dynamixel::GroupSyncWrite *groupSyncWritePID_;
    dynamixel::GroupSyncWrite *groupSyncWriteFF_;
    dynamixel::GroupBulkRead  *groupBulkReadState_;

    // 마지막으로 설정한 목표 위치 저장
    std::vector<int32_t> last_goal_position_;

    // 내부 헬퍼 함수
    void initGroupHandlers();
    void clearGroupHandlers();

    // 컨트롤 테이블 주소 (X-Series 기준)
    static constexpr uint16_t ADDR_BAUDRATE       = 8;
    static constexpr uint16_t ADDR_RETURN_DELAY   = 9;
    static constexpr uint16_t ADDR_DRIVE_MODE     = 10;
    static constexpr uint16_t ADDR_OPERATING_MODE = 11;
    static constexpr uint16_t ADDR_TORQUE_ENABLE  = 64;
    static constexpr uint16_t ADDR_POS_D_GAIN     = 80;
    static constexpr uint16_t ADDR_POS_I_GAIN     = 82;
    static constexpr uint16_t ADDR_POS_P_GAIN     = 84;
    static constexpr uint16_t ADDR_FF2_GAIN       = 88;
    static constexpr uint16_t ADDR_FF1_GAIN       = 90;
    static constexpr uint16_t ADDR_GOAL_VELOCITY  = 104;
    static constexpr uint16_t ADDR_PROFILE_ACC    = 108;
    static constexpr uint16_t ADDR_PROFILE_VEL    = 112;
    static constexpr uint16_t ADDR_GOAL_POSITION  = 116;

    static constexpr uint16_t ADDR_PRESENT_CURRENT  = 126;
    static constexpr uint16_t ADDR_PRESENT_VELOCITY = 128;
    static constexpr uint16_t ADDR_PRESENT_POSITION = 132;
    static constexpr uint16_t ADDR_PRESENT_VOLTAGE  = 144;

public:
    // --- 생성자 및 소멸자 ---
    DynamixelDriver(const std::string& device_name, float protocol_version, const std::vector<uint8_t>& ids);
    ~DynamixelDriver();

    // 복사 방지
    DynamixelDriver(const DynamixelDriver&) = delete;
    DynamixelDriver& operator=(const DynamixelDriver&) = delete;


    // --- 연결 및 초기화 ---
    bool connect(int target_baudrate);
    void disconnect();
    bool ping();

    // 모터 설정
    bool setOperatingMode(uint8_t mode); // 1(Vel), 3(Pos), 4(Ext Pos), 16(PWM)
    bool setDriveMode(bool is_time_based); // true (Time-based Profile), false (Velocity-based Profile)
    bool setReturnDelayTime(uint8_t delay_time); // Return Delay Time 설정
    bool setTorque(bool enable); // Torque On/Off
    bool reboot(); // 에러 시 리부팅


    // --- 튜닝 (PID & Profile) ---
    bool setPositionPID(const std::vector<uint16_t>& p_gains, const std::vector<uint16_t>& i_gains, const std::vector<uint16_t>& d_gains);
    bool setFeedforward(const std::vector<uint16_t>& ff1_gains, const std::vector<uint16_t>& ff2_gains);
    bool setProfile(uint32_t velocity, uint32_t acceleration);


    // --- 구동 (Write) ---
    bool writeGoalPosition(const std::vector<int32_t>& position);
    bool writeGoalVelocity(const std::vector<int32_t>& velocity);


    // --- 상태 읽기 (Read) ---
    bool readAllState(std::vector<MotorState>& out_states);
    bool readPresentCurrent(std::vector<int16_t>& out_currents);
    bool readPresentVoltage(std::vector<uint16_t>& out_voltages);

    std::vector<int32_t> getLastGoalPosition() const {
        return last_goal_position_;
    }
};

#endif