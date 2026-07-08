#ifndef CONFIG_H
#define CONFIG_H

#include <toml++/toml.h>
#include <string>
#include <vector>
#include <iostream>
#include <cstdint>
#include <cstdlib>
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

struct WebSocketConfig {
    int port = 9200;
};

struct RobotConfig {
    int32_t default_pitch;
    int32_t default_roll_r;
    int32_t default_roll_l;
    int32_t default_yaw;
    int32_t default_mouth;
    int led_pwm_pin;            // LED 밝기 PWM용 BCM GPIO 번호. -1 = 비활성
    int led_pwm_range;          // softPwm duty 범위 (0..range)

    double pulley_diameter;
    double height;
    double hole_radius;
    double yaw_gear_ratio;
    double mouth_back_compensation;
    double mouth_pitch_compensation;
    double max_mouth;
    double min_mouth;

    // ===== mouth↔roll zero-sum 실험 (임시) =====
    // experiment=true면 입 표현 budget을 입 모터와 roll R/L로 ratio 분배(합 100%).
    // false(기본)면 기존 거동 그대로. 끄려면 mouth_roll_experiment=false.
    bool   mouth_roll_experiment = false;  // 실험 on/off
    double mouth_roll_ratio = 0.0;         // roll R/L 분담 r (0=입 모터 100%, 1=roll 100%)
    double mouth_roll_gain = 0.75;         // roll 분담분 mouth tick→roll tick 환산 gain

    bool generate_head_motion;
    double wait_mode_rpy_ratio;
    double control_motor_rpy_ratio;

    // yaw(ID4) 물리 안전 한계 — 절대 넘으면 안 되는 tick 범위. 모든 yaw 명령에 항상 클램프 적용.
    // 하드 한계 ≈ 3070(좌45°)~4060(우45°), 홈 3600. 안전 마진 기본 3200~4000.
    int32_t yaw_tick_min = 3200;
    int32_t yaw_tick_max = 4000;
};

// ReSpeaker DOA 기반 고개(yaw) 추적 (true=활성, false=비활성). 없으면 false.
struct HeadTrackingConfig {
    bool enabled = false;
};

// 전역 인스턴스
inline WebSocketConfig cfg_ws;
inline DynamixelConfig cfg_dxl;
inline RobotConfig cfg_robot;
inline HeadTrackingConfig cfg_ht;


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
            auto val = arr->get(i)->template value<ValType>();
            if (!val) {
                std::cerr << "[Config Error] '" << key << "' 배열의 " << i << "번 인덱스 값이 잘못되었습니다.\n";
                return false;
            }
            dest_vec.push_back(*val);
        }
        return true;
    };

    // 데이터 로드 및 검증 (실패 시 즉시 종료)
    
    // [websocket] 섹션 (옵션 — 없으면 기본값 사용)
    if (tbl["websocket"].is_table()) {
        auto ws_node = tbl["websocket"];
        REQ(ws_node, "port", cfg_ws.port);
    }

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

    // 기기별 모터 초기 위치: RAY_UNIT 환경변수로 [robot.unitN] 섹션 선택.
    // 잘못된 초기값으로 홈 이동하면 하드웨어가 위험하므로 미설정/오타는 즉시 실패.
    const char* unit = std::getenv("RAY_UNIT");
    if (unit == nullptr || *unit == '\0') {
        std::cerr << "[Config Error] RAY_UNIT 환경변수가 설정되지 않았습니다 (예: RAY_UNIT=unit1).\n";
        return false;
    }
    if (!robot_node[unit].is_table()) {
        std::cerr << "[Config Error] [robot." << unit << "] 섹션이 config.toml에 없습니다.\n";
        return false;
    }
    auto unit_node = robot_node[unit];
    std::cout << "[Config] 기기 설정 선택: [robot." << unit << "]" << std::endl;

    ok &= REQ(unit_node, "default_pitch",  cfg_robot.default_pitch);
    ok &= REQ(unit_node, "default_roll_r", cfg_robot.default_roll_r);
    ok &= REQ(unit_node, "default_roll_l", cfg_robot.default_roll_l);
    ok &= REQ(unit_node, "default_yaw",    cfg_robot.default_yaw);
    ok &= REQ(unit_node, "default_mouth",  cfg_robot.default_mouth);
    ok &= REQ(robot_node, "led_pwm_pin",       cfg_robot.led_pwm_pin);
    ok &= REQ(robot_node, "led_pwm_range",     cfg_robot.led_pwm_range);

    ok &= REQ(robot_node, "pulley_diameter", cfg_robot.pulley_diameter);
    ok &= REQ(robot_node, "height",          cfg_robot.height);
    ok &= REQ(robot_node, "hole_radius",     cfg_robot.hole_radius);
    ok &= REQ(robot_node, "yaw_gear_ratio",  cfg_robot.yaw_gear_ratio);
    ok &= REQ(robot_node, "mouth_back_compensation",  cfg_robot.mouth_back_compensation);
    ok &= REQ(robot_node, "mouth_pitch_compensation", cfg_robot.mouth_pitch_compensation);
    ok &= REQ(robot_node, "max_mouth",      cfg_robot.max_mouth);
    ok &= REQ(robot_node, "min_mouth",      cfg_robot.min_mouth);

    ok &= REQ(robot_node, "generate_head_motion",    cfg_robot.generate_head_motion);
    ok &= REQ(robot_node, "wait_mode_rpy_ratio",     cfg_robot.wait_mode_rpy_ratio);
    ok &= REQ(robot_node, "control_motor_rpy_ratio",  cfg_robot.control_motor_rpy_ratio);

    if (!ok) return false;

    // yaw 안전 한계 (옵션 — 없으면 기본 3200~4000 유지)
    if (auto v = robot_node["yaw_tick_min"].value<int32_t>()) cfg_robot.yaw_tick_min = *v;
    if (auto v = robot_node["yaw_tick_max"].value<int32_t>()) cfg_robot.yaw_tick_max = *v;

    // mouth↔roll zero-sum 실험 (옵션 — 없으면 비활성, 기존 거동 유지)
    if (auto v = robot_node["mouth_roll_experiment"].value<bool>())  cfg_robot.mouth_roll_experiment = *v;
    if (auto v = robot_node["mouth_roll_ratio"].value<double>())     cfg_robot.mouth_roll_ratio = *v;
    if (auto v = robot_node["mouth_roll_gain"].value<double>())      cfg_robot.mouth_roll_gain = *v;

    // [head_tracking] 섹션 (옵션 — 없으면 비활성)
    if (tbl["head_tracking"].is_table()) {
        if (auto v = tbl["head_tracking"]["enabled"].value<bool>()) cfg_ht.enabled = *v;
    }
    std::cout << "[Config] head_tracking.enabled = " << (cfg_ht.enabled ? "true" : "false")
              << ", yaw 한계 tick = [" << cfg_robot.yaw_tick_min << ", " << cfg_robot.yaw_tick_max << "]\n";

    std::cout << "설정 파일 로드 완료.\n";
    return true;
}

#endif