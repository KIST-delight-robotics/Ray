#ifndef CONFIG_H
#define CONFIG_H

#include <toml++/toml.h>
#include <string>
#include <vector>
#include <iostream>
#include <cstdint>
#include <optional>

struct DynamixelConfig {
    float protocol_version;
    std::string device_name;
    std::vector<uint8_t> ids;
    int baudrate;
    bool is_time_based;
    uint8_t operating_mode;
    uint8_t return_delay_time;
    uint32_t profile_velocity_homing;
    uint32_t profile_velocity;
    uint32_t profile_acceleration;
    std::vector<uint16_t> pos_p_gain;
    std::vector<uint16_t> pos_i_gain;
    std::vector<uint16_t> pos_d_gain;
};

struct RobotConfig {
    int32_t default_pitch;
    int32_t default_roll_r;
    int32_t default_roll_l;
    int32_t default_yaw;
    int32_t default_mouth;

    double pulley_diameter;
    double height;
    double hole_radius;
    double yaw_gear_ratio;
    double mouth_tune;
    double mouth_back_compensation;
    double mouth_pitch_compensation;
};

// 전역 인스턴스
inline DynamixelConfig cfg_dxl;
inline RobotConfig cfg_robot;


inline bool LoadConfig(const std::string& path = "config.toml") {
    toml::table tbl;
    
    try {
        tbl = toml::parse_file(path);
    } catch (const toml::parse_error& err) {
        std::cerr << "[Config Error] 파일 파싱 실패: " << err << "\n";
        return false;
    }
    
    // 단일 값 읽기
    auto REQ = [](const toml::node_view<toml::node>& node, const char* key, auto& dest) -> bool {
        auto val = node[key].value<std::decay_t<decltype(dest)>>(); // 대상 변수 타입으로 읽기 시도
        if (!val) {
            std::cerr << "[Config Error] '" << key << "' 값이 없거나 타입이 잘못되었습니다.\n";
            return false;
        }
        dest = *val;
        return true;
    };

    // 벡터 읽기
    auto REQ_VEC = [](const toml::node_view<toml::node>& node, const char* key, auto& dest_vec) -> bool {
        auto* arr = node[key].as_array();
        if (!arr) {
            std::cerr << "[Config Error] '" << key << "' 배열이 없거나 형식이 잘못되었습니다.\n";
            return false;
        }
        
        using ValType = typename std::decay_t<decltype(dest_vec)>::value_type; // 벡터 내부 타입 추론
        dest_vec.clear();
        
        for (size_t i = 0; i < arr->size(); i++) {
            auto val = arr->get(i)->value<ValType>();
            if (!val) {
                std::cerr << "[Config Error] '" << key << "' 배열의 " << i << "번 인덱스 값이 잘못되었습니다.\n";
                return false;
            }
            dest_vec.push_back(*val);
        }
        return true;
    };

    // 데이터 로드 및 검증 (실패 시 즉시 종료)
    
    // [dynamixel] 섹션 확인
    if (!tbl["dynamixel"].is_table()) {
        std::cerr << "[Config Error] [dynamixel] 섹션이 없습니다.\n";
        return false;
    }
    auto dxl_node = tbl["dynamixel"];

    bool ok = true;
    
    ok &= REQ(dxl_node, "protocol_version", cfg_dxl.protocol_version);
    ok &= REQ(dxl_node, "device_name",      cfg_dxl.device_name);
    ok &= REQ_VEC(dxl_node, "ids",          cfg_dxl.ids);
    ok &= REQ(dxl_node, "baudrate",         cfg_dxl.baudrate);
    ok &= REQ(dxl_node, "is_time_based",    cfg_dxl.is_time_based);
    ok &= REQ(dxl_node, "operating_mode",   cfg_dxl.operating_mode);
    ok &= REQ(dxl_node, "return_delay_time",cfg_dxl.return_delay_time);
    ok &= REQ(dxl_node, "profile_velocity_homing", cfg_dxl.profile_velocity_homing);
    ok &= REQ(dxl_node, "profile_velocity",     cfg_dxl.profile_velocity);
    ok &= REQ(dxl_node, "profile_acceleration", cfg_dxl.profile_acceleration);
    ok &= REQ_VEC(dxl_node, "pos_p_gain",   cfg_dxl.pos_p_gain);
    ok &= REQ_VEC(dxl_node, "pos_i_gain",   cfg_dxl.pos_i_gain);
    ok &= REQ_VEC(dxl_node, "pos_d_gain",   cfg_dxl.pos_d_gain);

    if (!ok) return false;

    // [robot] 섹션 확인
    if (!tbl["robot"].is_table()) {
        std::cerr << "[Config Error] [robot] 섹션이 없습니다.\n";
        return false;
    }
    auto robot_node = tbl["robot"];

    ok &= REQ(robot_node, "default_pitch",  cfg_robot.default_pitch);
    ok &= REQ(robot_node, "default_roll_r", cfg_robot.default_roll_r);
    ok &= REQ(robot_node, "default_roll_l", cfg_robot.default_roll_l);
    ok &= REQ(robot_node, "default_yaw",    cfg_robot.default_yaw);
    ok &= REQ(robot_node, "default_mouth",  cfg_robot.default_mouth);

    ok &= REQ(robot_node, "pulley_diameter", cfg_robot.pulley_diameter);
    ok &= REQ(robot_node, "height",          cfg_robot.height);
    ok &= REQ(robot_node, "hole_radius",     cfg_robot.hole_radius);
    ok &= REQ(robot_node, "yaw_gear_ratio",  cfg_robot.yaw_gear_ratio);
    ok &= REQ(robot_node, "mouth_tune",      cfg_robot.mouth_tune);
    ok &= REQ(robot_node, "mouth_back_compensation",  cfg_robot.mouth_back_compensation);
    ok &= REQ(robot_node, "mouth_pitch_compensation", cfg_robot.mouth_pitch_compensation);

    if (!ok) return false;

    std::cout << "설정 파일 로드 완료.\n";
    return true;
}

#endif