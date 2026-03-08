#include "DynamixelDriver.h"
#include <iostream>
#include <algorithm>
#include <thread>
#include <chrono>

// SDK 매크로가 없는 경우를 대비한 정의 (보통 dynamixel_sdk.h에 포함됨)
#ifndef COMM_SUCCESS
#define COMM_SUCCESS 0
#endif

#ifndef BROADCAST_ID
#define BROADCAST_ID 0xFE
#endif

// 바이트 변환 매크로 (Little Endian) (dynamixel_sdk.h에 포함됨)
// #define DXL_LOBYTE(w)           ((uint8_t)(((uint64_t)(w)) & 0xff))
// #define DXL_HIBYTE(w)           ((uint8_t)(((uint64_t)(w) >> 8) & 0xff))
// #define DXL_LOWORD(l)           ((uint16_t)(((uint64_t)(l)) & 0xffff))
// #define DXL_HIWORD(l)           ((uint16_t)(((uint64_t)(l) >> 16) & 0xffff))

DynamixelDriver::DynamixelDriver(const std::string& device_name, float protocol_version, const std::vector<uint8_t>& ids)
    : device_name_(device_name), protocol_version_(protocol_version), dxl_ids_(ids),
      portHandler_(nullptr), packetHandler_(nullptr),
      groupSyncWritePos_(nullptr), groupSyncWriteVel_(nullptr), groupSyncWritePID_(nullptr), groupSyncWriteFF_(nullptr), groupBulkReadState_(nullptr)
{
    // Handler 생성
    portHandler_ = dynamixel::PortHandler::getPortHandler(device_name_.c_str());
    packetHandler_ = dynamixel::PacketHandler::getPacketHandler(protocol_version_);
}

DynamixelDriver::~DynamixelDriver() {
    disconnect();
    
    // Handler 삭제
    if (portHandler_) delete portHandler_;
    if (packetHandler_) delete packetHandler_;
}

void DynamixelDriver::initGroupHandlers() {
    clearGroupHandlers();

    groupSyncWritePos_ = new dynamixel::GroupSyncWrite(portHandler_, packetHandler_, ADDR_GOAL_POSITION, 4);
    groupSyncWriteVel_ = new dynamixel::GroupSyncWrite(portHandler_, packetHandler_, ADDR_GOAL_VELOCITY, 4);
    groupSyncWritePID_ = new dynamixel::GroupSyncWrite(portHandler_, packetHandler_, ADDR_POS_D_GAIN, 6); // D, I, P 각 2바이트씩 총 6바이트
    groupSyncWriteFF_ = new dynamixel::GroupSyncWrite(portHandler_, packetHandler_, ADDR_FF2_GAIN, 4); // FF1, FF2 각 2바이트씩 총 4바이트

    groupBulkReadState_ = new dynamixel::GroupBulkRead(portHandler_, packetHandler_);
    for (uint8_t id : dxl_ids_) {
        // Addr 126부터 10바이트 읽기 (현재 위치, 속도, 전류)
        if (!groupBulkReadState_->addParam(id, ADDR_PRESENT_CURRENT, 10)) {
            std::cerr << "[DynamixelDriver] Failed to add BulkRead param for ID " << (int)id << std::endl;
        }
    }
}

void DynamixelDriver::clearGroupHandlers() {
    if (groupSyncWritePos_) { delete groupSyncWritePos_; groupSyncWritePos_ = nullptr; }
    if (groupSyncWriteVel_) { delete groupSyncWriteVel_; groupSyncWriteVel_ = nullptr; }
    if (groupSyncWritePID_) { delete groupSyncWritePID_; groupSyncWritePID_ = nullptr; }
    if (groupSyncWriteFF_) { delete groupSyncWriteFF_; groupSyncWriteFF_ = nullptr; }
    if (groupBulkReadState_) { delete groupBulkReadState_; groupBulkReadState_ = nullptr; }
}

bool DynamixelDriver::connect(int target_baudrate) {
    std::lock_guard<std::mutex> lock(dxl_mutex_);

    if (!portHandler_->openPort()) {
        std::cerr << "[DynamixelDriver] Failed to open port: " << device_name_ << std::endl;
        return false;
    }

    // 1. 먼저 Target Baudrate로 시도
    if (portHandler_->setBaudRate(target_baudrate)) {
        uint8_t error = 0;
        // 첫 번째 ID로 핑 테스트
        if (dxl_ids_.empty() || packetHandler_->ping(portHandler_, dxl_ids_[0], &error) == COMM_SUCCESS) {
            std::cout << "[DynamixelDriver] Connected at " << target_baudrate << " bps." << std::endl;
            initGroupHandlers();
            return true;
        }
    }

    // 2. 실패 시 스캔 시작
    std::cout << "[DynamixelDriver] Target baudrate failed. Scanning..." << std::endl;
    std::vector<int> baudrates = {57600, 115200, 1000000, 2000000, 3000000, 4000000, 4500000};
    
    bool found = false;
    for (int baud : baudrates) {
        if (baud == target_baudrate) continue; // 이미 해봄

        if (portHandler_->setBaudRate(baud)) {
            uint8_t error = 0;
            // 첫 번째 ID로 핑 테스트
            if (!dxl_ids_.empty() && packetHandler_->ping(portHandler_, dxl_ids_[0], &error) == COMM_SUCCESS) {
                std::cout << "[DynamixelDriver] Found motors at " << baud << " bps. Configuring to " << target_baudrate << "..." << std::endl;
                
                // 3. 찾았으면 Target Baudrate로 변경 명령 전송
                // Baudrate 값 매핑
                uint8_t baud_val = 0;
                switch(target_baudrate) {
                    case 9600: baud_val = 0; break;
                    case 57600: baud_val = 1; break;
                    case 115200: baud_val = 2; break;
                    case 1000000: baud_val = 3; break;
                    case 2000000: baud_val = 4; break;
                    case 3000000: baud_val = 5; break;
                    case 4000000: baud_val = 6; break;
                    case 4500000: baud_val = 7; break;
                    default: 
                        std::cerr << "Unsupported target baudrate." << std::endl;
                        return false;
                }

                // 모든 모터의 보드레이트 변경 (Broadcast ID 사용)
                packetHandler_->write1ByteTxOnly(portHandler_, BROADCAST_ID, ADDR_TORQUE_ENABLE, 0); // 토크 끄기
                packetHandler_->write1ByteTxOnly(portHandler_, BROADCAST_ID, ADDR_BAUDRATE, baud_val); // 보드레이트 변경
                
                // PC 포트 변경
                portHandler_->setBaudRate(target_baudrate);

                std::this_thread::sleep_for(std::chrono::milliseconds(100));
                found = true;
                break;
            }
        }
    }

    if (found) {
        std::cout << "[DynamixelDriver] Successfully reconfigured to " << target_baudrate << " bps." << std::endl;
        initGroupHandlers();
        return true;
    } else {
        std::cerr << "[DynamixelDriver] Failed to find motors." << std::endl;
        portHandler_->closePort();
        return false;
    }
}

void DynamixelDriver::disconnect() {
    std::lock_guard<std::mutex> lock(dxl_mutex_);
    
    clearGroupHandlers(); // 메모리 해제

    if (portHandler_) {
        // 토크 끄고 포트 닫기
        packetHandler_->write1ByteTxOnly(portHandler_, BROADCAST_ID, ADDR_TORQUE_ENABLE, 0);
        portHandler_->closePort();
        std::cout << "[DynamixelDriver] Port closed." << std::endl;
    }
}

bool DynamixelDriver::ping() {
    std::lock_guard<std::mutex> lock(dxl_mutex_);
    bool all_success = true;
    uint8_t error = 0;

    for (uint8_t id : dxl_ids_) {
        if (packetHandler_->ping(portHandler_, id, &error) != COMM_SUCCESS) {
            std::cerr << "[DynamixelDriver] Failed to ping ID " << (int)id << std::endl;
            all_success = false;
        }
    }
    return all_success;
}

// --- 모터 설정 ---
bool DynamixelDriver::setDriveMode(bool is_time_based) {
    std::lock_guard<std::mutex> lock(dxl_mutex_);
    bool result = true;
    uint8_t error = 0;

    packetHandler_->write1ByteTxOnly(portHandler_, BROADCAST_ID, ADDR_TORQUE_ENABLE, 0); // 토크 끄기
    
    // Drive Mode 설정
    // Bit 2: 0=Velocity-based, 1=Time-based
    // Bit 0: Normal/Reverse (이건 건드리면 안됨)
    for (uint8_t id : dxl_ids_) {
        uint8_t current_val = 0;
        if (packetHandler_->read1ByteTxRx(portHandler_, id, ADDR_DRIVE_MODE, &current_val, &error) == COMM_SUCCESS) {
            uint8_t new_val = current_val;
            if (is_time_based) new_val |= 0x04;  // Set Bit 2
            else               new_val &= ~0x04; // Clear Bit 2
            
            if (new_val != current_val) {
                if (packetHandler_->write1ByteTxRx(portHandler_, id, ADDR_DRIVE_MODE, new_val, &error) != COMM_SUCCESS) {
                    result = false;
                }
            }
        } else {
            result = false;
        }
    }
    return result;
}

bool DynamixelDriver::setOperatingMode(uint8_t mode) {
    std::lock_guard<std::mutex> lock(dxl_mutex_);
    bool result = true;
    uint8_t error = 0;

    packetHandler_->write1ByteTxOnly(portHandler_, BROADCAST_ID, ADDR_TORQUE_ENABLE, 0); // 토크 끄기

    for (uint8_t id : dxl_ids_) {
        uint8_t current_mode = 0;
        // 현재 모드 읽기
        if (packetHandler_->read1ByteTxRx(portHandler_, id, ADDR_OPERATING_MODE, &current_mode, &error) == COMM_SUCCESS) {
            if (current_mode != mode) {
                if (packetHandler_->write1ByteTxRx(portHandler_, id, ADDR_OPERATING_MODE, mode, &error) != COMM_SUCCESS) {
                    std::cerr << "[DynamixelDriver] Failed to set Operating Mode for ID " << (int)id << std::endl;
                    result = false;
                }
            }
        } else {
            std::cerr << "[DynamixelDriver] Failed to read Operating Mode for ID " << (int)id << std::endl;
            if (packetHandler_->write1ByteTxRx(portHandler_, id, ADDR_OPERATING_MODE, mode, &error) != COMM_SUCCESS) {
                std::cerr << "[DynamixelDriver] Failed to set Operating Mode for ID " << (int)id << std::endl;
                result = false;
            }
        }
    }
    return result;
}

bool DynamixelDriver::setReturnDelayTime(uint8_t delay_time) {
    std::lock_guard<std::mutex> lock(dxl_mutex_);
    bool result = true;
    uint8_t error = 0;

    packetHandler_->write1ByteTxOnly(portHandler_, BROADCAST_ID, ADDR_TORQUE_ENABLE, 0); // 토크 끄기

    for (uint8_t id : dxl_ids_) {
        uint8_t current_val = 0;
        
        // 현재 설정값 읽기
        if (packetHandler_->read1ByteTxRx(portHandler_, id, ADDR_RETURN_DELAY, &current_val, &error) == COMM_SUCCESS) {
            
            // 값이 다를 때만 쓰기 수행
            if (current_val != delay_time) {
                if (packetHandler_->write1ByteTxRx(portHandler_, id, ADDR_RETURN_DELAY, delay_time, &error) != COMM_SUCCESS) {
                    std::cerr << "[DynamixelDriver] Failed to set Return Delay Time for ID " << (int)id << std::endl;
                    result = false;
                } else {
                    std::cout << "[DynamixelDriver] ID " << (int)id << " Return Delay Time changed: " 
                              << (int)current_val << " -> " << (int)delay_time << std::endl;
                }
            }
        } else {
            // 읽기 실패 시 에러 출력
            std::cerr << "[DynamixelDriver] Failed to read Return Delay Time for ID " << (int)id << std::endl;
            // 설정 시도
            if (packetHandler_->write1ByteTxRx(portHandler_, id, ADDR_RETURN_DELAY, delay_time, &error) != COMM_SUCCESS) {
                std::cerr << "[DynamixelDriver] Failed to set Return Delay Time for ID " << (int)id << std::endl;
                result = false;
            }
        }
    }
    return result;
}

bool DynamixelDriver::setTorque(bool enable) {
    std::lock_guard<std::mutex> lock(dxl_mutex_);
    bool result = true;
    uint8_t error = 0;
    uint8_t val = enable ? 1 : 0;

    for (uint8_t id : dxl_ids_) {
        if (packetHandler_->write1ByteTxRx(portHandler_, id, ADDR_TORQUE_ENABLE, val, &error) != COMM_SUCCESS) {
            std::cerr << "[DynamixelDriver] Failed to set Torque for ID " << (int)id << std::endl;
            result = false;
        }
    }
    return result;
}

bool DynamixelDriver::reboot() {
    std::lock_guard<std::mutex> lock(dxl_mutex_);
    bool result = true;
    uint8_t error = 0;

    for (uint8_t id : dxl_ids_) {
        if (packetHandler_->reboot(portHandler_, id, &error) != COMM_SUCCESS) {
            std::cerr << "[DynamixelDriver] Failed to reboot ID " << (int)id << std::endl;
            result = false;
        }
    }
    return result;
}

// --- 튜닝 ---

bool DynamixelDriver::setPositionPID(const std::vector<uint16_t>& p_gains, const std::vector<uint16_t>& i_gains, const std::vector<uint16_t>& d_gains) {
    std::lock_guard<std::mutex> lock(dxl_mutex_);
    
    // 사이즈 체크
    size_t num_motors = dxl_ids_.size();
    if (p_gains.size() != num_motors || i_gains.size() != num_motors || d_gains.size() != num_motors) {
        std::cerr << "[DynamixelDriver] PID Gain vector size mismatch!" << std::endl;
        return false;
    }

    groupSyncWritePID_->clearParam();
    uint8_t param[6]; // D(2) + I(2) + P(2)

    for (size_t i = 0; i < num_motors; i++) {
        // X-Series 메모리 순서: D(80) -> I(82) -> P(84)
        
        // D Gain
        param[0] = DXL_LOBYTE(d_gains[i]);
        param[1] = DXL_HIBYTE(d_gains[i]);
        
        // I Gain
        param[2] = DXL_LOBYTE(i_gains[i]);
        param[3] = DXL_HIBYTE(i_gains[i]);
        
        // P Gain
        param[4] = DXL_LOBYTE(p_gains[i]);
        param[5] = DXL_HIBYTE(p_gains[i]);

        if (!groupSyncWritePID_->addParam(dxl_ids_[i], param)) {
            return false;
        }
    }

    if (groupSyncWritePID_->txPacket() != COMM_SUCCESS) {
        std::cerr << "[DynamixelDriver] Failed to set PID Gains via SyncWrite." << std::endl;
        return false;
    }

    return true;
}

bool DynamixelDriver::setFeedforward(const std::vector<uint16_t>& ff1_gains, const std::vector<uint16_t>& ff2_gains) {
    std::lock_guard<std::mutex> lock(dxl_mutex_);
    
    size_t num_motors = dxl_ids_.size();
    if (ff1_gains.size() != num_motors || ff2_gains.size() != num_motors) {
        std::cerr << "[DynamixelDriver] FF Gain vector size mismatch!" << std::endl;
        return false;
    }

    groupSyncWriteFF_->clearParam();
    uint8_t param[4]; // FF2(2) + FF1(2)

    for (size_t i = 0; i < num_motors; i++) {
        // X-Series 메모리 순서: FF2(88) -> FF1(90)
        
        // FF2 Gain (Acceleration)
        param[0] = DXL_LOBYTE(ff2_gains[i]);
        param[1] = DXL_HIBYTE(ff2_gains[i]);
        
        // FF1 Gain (Velocity)
        param[2] = DXL_LOBYTE(ff1_gains[i]);
        param[3] = DXL_HIBYTE(ff1_gains[i]);

        if (!groupSyncWriteFF_->addParam(dxl_ids_[i], param)) {
            return false;
        }
    }

    if (groupSyncWriteFF_->txPacket() != COMM_SUCCESS) {
        std::cerr << "[DynamixelDriver] Failed to set FF Gains via SyncWrite." << std::endl;
        return false;
    }
    
    return true;
}

bool DynamixelDriver::setProfile(uint32_t velocity, uint32_t acceleration) {
    std::lock_guard<std::mutex> lock(dxl_mutex_);
    bool result = true;
    uint8_t error = 0;

    for (uint8_t id : dxl_ids_) {
        if (packetHandler_->write4ByteTxRx(portHandler_, id, ADDR_PROFILE_VEL, velocity, &error) != COMM_SUCCESS) result = false;
        if (packetHandler_->write4ByteTxRx(portHandler_, id, ADDR_PROFILE_ACC, acceleration, &error) != COMM_SUCCESS) result = false;
    }
    return result;
}

// --- 구동 ---

bool DynamixelDriver::writeGoalPosition(const std::vector<int32_t>& position) {
    std::lock_guard<std::mutex> lock(dxl_mutex_);
    
    if (position.size() != dxl_ids_.size()) {
        std::cerr << "[DynamixelDriver] Position count mismatch!" << std::endl;
        return false;
    }

    groupSyncWritePos_->clearParam();
    uint8_t param_goal_position[4];

    for (size_t i = 0; i < dxl_ids_.size(); i++) {
        // int32 -> byte array conversion
        param_goal_position[0] = DXL_LOBYTE(DXL_LOWORD(position[i]));
        param_goal_position[1] = DXL_HIBYTE(DXL_LOWORD(position[i]));
        param_goal_position[2] = DXL_LOBYTE(DXL_HIWORD(position[i]));
        param_goal_position[3] = DXL_HIBYTE(DXL_HIWORD(position[i]));
        if (!groupSyncWritePos_->addParam(dxl_ids_[i], param_goal_position)) {
            return false;
        }
    }

    int dxl_comm_result = groupSyncWritePos_->txPacket();
    if (dxl_comm_result != COMM_SUCCESS) {
        std::cerr << "[DynamixelDriver] SyncWrite Position Failed: " << packetHandler_->getTxRxResult(dxl_comm_result) << std::endl;
        return false;
    }

    // 마지막 위치 업데이트
    last_goal_position_ = position; 

    return true;
}

bool DynamixelDriver::writeGoalVelocity(const std::vector<int32_t>& velocity) {
    std::lock_guard<std::mutex> lock(dxl_mutex_);

    if (velocity.size() != dxl_ids_.size()) return false;

    groupSyncWriteVel_->clearParam();
    uint8_t param_goal_velocity[4];

    for (size_t i = 0; i < dxl_ids_.size(); i++) {
        param_goal_velocity[0] = DXL_LOBYTE(DXL_LOWORD(velocity[i]));
        param_goal_velocity[1] = DXL_HIBYTE(DXL_LOWORD(velocity[i]));
        param_goal_velocity[2] = DXL_LOBYTE(DXL_HIWORD(velocity[i]));
        param_goal_velocity[3] = DXL_HIBYTE(DXL_HIWORD(velocity[i]));

        if (!groupSyncWriteVel_->addParam(dxl_ids_[i], param_goal_velocity)) {
            return false;
        }
    }

    if (groupSyncWriteVel_->txPacket() != COMM_SUCCESS) return false;
    return true;
}

// --- 상태 읽기 ---

bool DynamixelDriver::readAllState(std::vector<MotorState>& out_states) {
    std::lock_guard<std::mutex> lock(dxl_mutex_);

    // BulkRead 실행
    int dxl_comm_result = groupBulkReadState_->txRxPacket();
    if (dxl_comm_result != COMM_SUCCESS) {
        std::cerr << "[DynamixelDriver] BulkRead Failed: " << packetHandler_->getTxRxResult(dxl_comm_result) << std::endl;
        return false;
    }

    // 결과 파싱
    out_states.clear();
    out_states.reserve(dxl_ids_.size());

    for (uint8_t id : dxl_ids_) {
        // 데이터가 도착했는지 확인 (ADDR_PRESENT_CURRENT, 10 bytes)
        if (groupBulkReadState_->isAvailable(id, ADDR_PRESENT_CURRENT, 10)) {
            MotorState state;
            
            // Current (2 bytes) at 126
            state.current = (int16_t)groupBulkReadState_->getData(id, ADDR_PRESENT_CURRENT, 2);
            
            // Velocity (4 bytes) at 128
            state.velocity = (int32_t)groupBulkReadState_->getData(id, ADDR_PRESENT_VELOCITY, 4);
            
            // Position (4 bytes) at 132
            state.position = (int32_t)groupBulkReadState_->getData(id, ADDR_PRESENT_POSITION, 4);

            out_states.push_back(state);
        } else {
            out_states.push_back({0, 0, 0}); // 에러 시 0
            std::cerr << "[DynamixelDriver] No data available for ID " << (int)id << std::endl;
        }
    }

    return true;
}

bool DynamixelDriver::readPresentCurrent(std::vector<int16_t>& out_currents) {
    std::lock_guard<std::mutex> lock(dxl_mutex_);
    
    out_currents.clear();
    uint8_t error = 0;
    int16_t current_val = 0;

    for (uint8_t id : dxl_ids_) {
        if (packetHandler_->read2ByteTxRx(portHandler_, id, ADDR_PRESENT_CURRENT, (uint16_t*)&current_val, &error) == COMM_SUCCESS) {
            out_currents.push_back(current_val);
        } else {
            out_currents.push_back(0); // 에러 시 0
            std::cerr << "[DynamixelDriver] Failed to read current for ID " << (int)id << std::endl;
        }
    }
    return true;
}

bool DynamixelDriver::readPresentVoltage(std::vector<uint16_t>& out_voltages) {
    std::lock_guard<std::mutex> lock(dxl_mutex_);

    out_voltages.clear();
    uint8_t error = 0;
    uint16_t voltage_val = 0;

    for (uint8_t id : dxl_ids_) {
        if (packetHandler_->read2ByteTxRx(portHandler_, id, ADDR_PRESENT_VOLTAGE, &voltage_val, &error) == COMM_SUCCESS) {
            out_voltages.push_back(voltage_val);
        } else {
            out_voltages.push_back(0);
        }
    }
    return true;
}