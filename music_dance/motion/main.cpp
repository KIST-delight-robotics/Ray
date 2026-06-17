// music_dance 모션 제어부.
//
// Python 분석부(analysis/analyze.py)가 만든 timeline.csv 를 읽어,
// WAV 를 재생하면서 재생 위치(단일 마스터 클럭)에 맞춰
//   - LED  : 라즈베리파이 하드웨어 PWM (sysfs /sys/class/pwm) 밝기 디밍
//   - 모터 : Dynamixel ID6 위치 제어 (DynamixelSDK)
// 를 동기 구동한다. 한 프로세스 · 한 클럭이라 LED·모터·오디오가 정합한다.
//
// 지금은 모터 1개(ID6)만 다룬다. 나중에 여러 모터로 확장 예정.
//
// 사용 예:
//   ./dance --timeline timeline.csv --wav ../../V_ZionT_MR.wav \
//           --port /dev/ttyUSB0 --baud 2000000 --id 6 \
//           --motor-home 100 --motor-amp 300 --pwmchip 0 --pwmchan 1
//
// 빌드: 같은 폴더 CMakeLists.txt 참조.

#include <alsa/asoundlib.h>
#include <sndfile.h>

#include <atomic>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "dynamixel_sdk.h"

// ===== Dynamixel X-시리즈 컨트롤 테이블 =====
namespace dxl_addr {
constexpr uint16_t TORQUE_ENABLE = 64;
constexpr uint16_t OPERATING_MODE = 11;
constexpr uint16_t PROFILE_ACC = 108;
constexpr uint16_t PROFILE_VEL = 112;
constexpr uint16_t GOAL_POSITION = 116;
constexpr uint16_t PRESENT_POSITION = 132;
}  // namespace dxl_addr

constexpr uint8_t OP_MODE_POSITION = 3;  // 0~4095 단일 회전 (안전)

// ===== 전역 종료 플래그 =====
static std::atomic<bool> g_stop{false};
static void on_sigint(int) { g_stop = true; }

// ===== 마스터 클럭 (재생 시작 시각) =====
static std::atomic<bool> g_playing{false};
static std::atomic<bool> g_player_done{false};  // 재생 스레드 종료(정상/실패) 표시
static std::chrono::steady_clock::time_point g_play_start;

// -------------------------------------------------------------------------
// 타임라인
// -------------------------------------------------------------------------
struct Timeline {
    double fps = 100.0;
    std::vector<float> led;
    std::vector<float> motor;

    bool load(const std::string& path) {
        std::ifstream f(path);
        if (!f) {
            fprintf(stderr, "[timeline] 열기 실패: %s\n", path.c_str());
            return false;
        }
        std::string line;
        bool header_done = false;
        while (std::getline(f, line)) {
            if (line.empty()) continue;
            if (line[0] == '#') {
                auto pos = line.find("fps=");
                if (pos != std::string::npos) fps = std::atof(line.c_str() + pos + 4);
                continue;
            }
            if (!header_done) {  // "led,motor" 컬럼 헤더
                header_done = true;
                continue;
            }
            float l = 0.f, m = 0.f;
            if (std::sscanf(line.c_str(), "%f,%f", &l, &m) == 2) {
                led.push_back(l);
                motor.push_back(m);
            }
        }
        return !led.empty();
    }

    size_t size() const { return led.size(); }
    double duration() const { return size() / fps; }
};

// -------------------------------------------------------------------------
// LED: sysfs 하드웨어 PWM
// -------------------------------------------------------------------------
class PwmLed {
public:
    bool open(int chip, int chan, long period_ns) {
        chip_ = chip;
        chan_ = chan;
        period_ns_ = period_ns;
        base_ = "/sys/class/pwm/pwmchip" + std::to_string(chip);
        ch_ = base_ + "/pwm" + std::to_string(chan);

        std::ifstream exists(ch_ + "/period");
        if (!exists.good()) {
            if (!write_str(base_ + "/export", std::to_string(chan))) {
                fprintf(stderr, "[pwm] export 실패 (root 권한 필요). LED 비활성.\n");
                return false;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(150));
        }
        // duty <= period 보장 위해 duty 0 먼저
        write_str(ch_ + "/duty_cycle", "0");
        if (!write_str(ch_ + "/period", std::to_string(period_ns_))) {
            fprintf(stderr, "[pwm] period 설정 실패. LED 비활성.\n");
            return false;
        }
        write_str(ch_ + "/polarity", "normal");  // 미지원 시 무시
        if (!write_str(ch_ + "/enable", "1")) {
            fprintf(stderr, "[pwm] enable 실패. LED 비활성.\n");
            return false;
        }
        ok_ = true;
        fprintf(stderr, "[pwm] OK: pwmchip%d/pwm%d, %.1fkHz\n", chip, chan, 1e6 / period_ns_);
        return true;
    }

    void set(float brightness) {
        if (!ok_) return;
        if (brightness < 0.f) brightness = 0.f;
        if (brightness > 1.f) brightness = 1.f;
        long duty = static_cast<long>(period_ns_ * brightness);
        write_str(ch_ + "/duty_cycle", std::to_string(duty));
    }

    void close() {
        if (!ok_) return;
        write_str(ch_ + "/duty_cycle", "0");
        write_str(ch_ + "/enable", "0");
        ok_ = false;
    }

    bool ok() const { return ok_; }

private:
    static bool write_str(const std::string& path, const std::string& val) {
        std::ofstream f(path);
        if (!f) return false;
        f << val;
        return f.good();
    }

    int chip_ = 0, chan_ = 1;
    long period_ns_ = 1000000;
    std::string base_, ch_;
    bool ok_ = false;
};

// -------------------------------------------------------------------------
// 모터: Dynamixel ID6
// -------------------------------------------------------------------------
class DxlMotor {
public:
    bool open(const std::string& port, int baud, uint8_t id, uint32_t prof_vel, uint32_t prof_acc) {
        id_ = id;
        port_ = dynamixel::PortHandler::getPortHandler(port.c_str());
        packet_ = dynamixel::PacketHandler::getPacketHandler(2.0);
        if (!port_->openPort()) {
            fprintf(stderr, "[dxl] 포트 열기 실패: %s. 모터 비활성.\n", port.c_str());
            return false;
        }
        if (!port_->setBaudRate(baud)) {
            fprintf(stderr, "[dxl] baud 설정 실패: %d. 모터 비활성.\n", baud);
            return false;
        }
        // 핑으로 존재 확인
        uint8_t err = 0;
        uint16_t model = 0;
        if (packet_->ping(port_, id_, &model, &err) != COMM_SUCCESS) {
            fprintf(stderr, "[dxl] ID %d 핑 실패 (baud/배선 확인). 모터 비활성.\n", id_);
            return false;
        }
        // 설정: 토크 끄고 → 모드/프로파일 → 토크 켜기
        write1(dxl_addr::TORQUE_ENABLE, 0);
        write1(dxl_addr::OPERATING_MODE, OP_MODE_POSITION);
        write4(dxl_addr::PROFILE_VEL, prof_vel);
        write4(dxl_addr::PROFILE_ACC, prof_acc);
        write1(dxl_addr::TORQUE_ENABLE, 1);
        ok_ = true;
        fprintf(stderr, "[dxl] OK: ID %d (model %d) @ %d baud\n", id_, model, baud);
        return true;
    }

    void moveTo(int32_t ticks) {
        if (!ok_) return;
        if (ticks < 0) ticks = 0;
        if (ticks > 4095) ticks = 4095;
        write4(dxl_addr::GOAL_POSITION, static_cast<uint32_t>(ticks));
    }

    void close(int32_t home_ticks) {
        if (!ok_) return;
        moveTo(home_ticks);
        std::this_thread::sleep_for(std::chrono::milliseconds(400));
        write1(dxl_addr::TORQUE_ENABLE, 0);
        port_->closePort();
        ok_ = false;
    }

    bool ok() const { return ok_; }

private:
    void write1(uint16_t addr, uint8_t v) {
        uint8_t err = 0;
        packet_->write1ByteTxRx(port_, id_, addr, v, &err);
    }
    void write4(uint16_t addr, uint32_t v) {
        uint8_t err = 0;
        packet_->write4ByteTxRx(port_, id_, addr, v, &err);
    }

    dynamixel::PortHandler* port_ = nullptr;
    dynamixel::PacketHandler* packet_ = nullptr;
    uint8_t id_ = 6;
    bool ok_ = false;
};

// -------------------------------------------------------------------------
// 오디오: libsndfile 로 로드 → ALSA 로 재생 (별도 스레드)
// -------------------------------------------------------------------------
struct AudioData {
    std::vector<float> samples;  // 인터리브
    int channels = 0;
    int rate = 0;
    sf_count_t frames = 0;
};

static bool load_wav(const std::string& path, AudioData& out) {
    SF_INFO info;
    std::memset(&info, 0, sizeof(info));
    SNDFILE* snd = sf_open(path.c_str(), SFM_READ, &info);
    if (!snd) {
        fprintf(stderr, "[audio] WAV 열기 실패: %s\n", path.c_str());
        return false;
    }
    out.channels = info.channels;
    out.rate = info.samplerate;
    out.frames = info.frames;
    out.samples.resize(static_cast<size_t>(info.frames) * info.channels);
    sf_readf_float(snd, out.samples.data(), info.frames);
    sf_close(snd);
    fprintf(stderr, "[audio] %s: %d Hz, %d ch, %.1f s\n", path.c_str(), out.rate, out.channels,
            static_cast<double>(out.frames) / out.rate);
    return true;
}

static void play_thread(const AudioData* a) {
    struct DoneGuard {
        ~DoneGuard() { g_player_done = true; }
    } done_guard;

    snd_pcm_t* pcm = nullptr;
    if (snd_pcm_open(&pcm, "default", SND_PCM_STREAM_PLAYBACK, 0) < 0) {
        fprintf(stderr, "[audio] ALSA 열기 실패\n");
        return;
    }
    if (snd_pcm_set_params(pcm, SND_PCM_FORMAT_FLOAT_LE, SND_PCM_ACCESS_RW_INTERLEAVED, a->channels,
                           a->rate, 1, 120000) < 0) {
        fprintf(stderr, "[audio] ALSA 파라미터 설정 실패\n");
        snd_pcm_close(pcm);
        return;
    }

    const snd_pcm_uframes_t chunk = 1024;
    sf_count_t pos = 0;
    bool started = false;
    while (!g_stop && pos < a->frames) {
        snd_pcm_uframes_t n = std::min<sf_count_t>(chunk, a->frames - pos);
        const float* buf = a->samples.data() + static_cast<size_t>(pos) * a->channels;
        if (!started) {  // 첫 write 직전 = 마스터 클럭 시작
            g_play_start = std::chrono::steady_clock::now();
            g_playing = true;
            started = true;
        }
        int rc = snd_pcm_writei(pcm, buf, n);
        if (rc == -EPIPE) {
            snd_pcm_prepare(pcm);  // underrun 복구
            continue;
        }
        if (rc < 0) {
            if (snd_pcm_recover(pcm, rc, 1) < 0) break;
            continue;
        }
        pos += rc;
    }
    if (!g_stop) snd_pcm_drain(pcm);
    snd_pcm_close(pcm);
    g_playing = false;
}

// -------------------------------------------------------------------------
// 인자 파싱
// -------------------------------------------------------------------------
struct Args {
    std::string timeline = "timeline.csv";
    std::string wav;
    std::string port = "/dev/ttyUSB0";
    int baud = 2000000;
    int id = 6;
    int motor_home = 100;
    int motor_amp = 300;
    int pwmchip = 0;
    int pwmchan = 1;
    double pwm_khz = 1.0;
    double sync_ms = 0.0;  // 양수면 시각효과를 늦춤 (오디오 버퍼 지연 보정)
    bool no_motor = false;
    bool no_led = false;
};

static Args parse_args(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string k = argv[i];
        auto next = [&]() -> std::string { return (i + 1 < argc) ? argv[++i] : ""; };
        if (k == "--timeline") a.timeline = next();
        else if (k == "--wav") a.wav = next();
        else if (k == "--port") a.port = next();
        else if (k == "--baud") a.baud = std::atoi(next().c_str());
        else if (k == "--id") a.id = std::atoi(next().c_str());
        else if (k == "--motor-home") a.motor_home = std::atoi(next().c_str());
        else if (k == "--motor-amp") a.motor_amp = std::atoi(next().c_str());
        else if (k == "--pwmchip") a.pwmchip = std::atoi(next().c_str());
        else if (k == "--pwmchan") a.pwmchan = std::atoi(next().c_str());
        else if (k == "--pwm-khz") a.pwm_khz = std::atof(next().c_str());
        else if (k == "--sync-ms") a.sync_ms = std::atof(next().c_str());
        else if (k == "--no-motor") a.no_motor = true;
        else if (k == "--no-led") a.no_led = true;
        else fprintf(stderr, "[args] 알 수 없는 옵션: %s\n", k.c_str());
    }
    return a;
}

// -------------------------------------------------------------------------
int main(int argc, char** argv) {
    Args args = parse_args(argc, argv);
    std::signal(SIGINT, on_sigint);
    std::signal(SIGTERM, on_sigint);

    if (args.wav.empty()) {
        fprintf(stderr, "사용법: --wav <path> [--timeline timeline.csv] [옵션]\n");
        return 1;
    }

    Timeline tl;
    if (!tl.load(args.timeline)) {
        fprintf(stderr, "[main] 타임라인 로드 실패\n");
        return 1;
    }
    fprintf(stderr, "[main] 타임라인: %zu frames @ %.2f fps (%.1f s)\n", tl.size(), tl.fps,
            tl.duration());

    AudioData audio;
    if (!load_wav(args.wav, audio)) return 1;

    PwmLed led;
    if (!args.no_led) led.open(args.pwmchip, args.pwmchan, static_cast<long>(1e6 / args.pwm_khz));

    DxlMotor motor;
    if (!args.no_motor) {
        if (motor.open(args.port, args.baud, static_cast<uint8_t>(args.id), 100, 20)) {
            motor.moveTo(args.motor_home);  // 홈으로
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
        }
    }

    fprintf(stderr, "[main] 재생 시작 (Ctrl-C 로 중단)\n");
    std::thread player(play_thread, &audio);

    // 재생 시작 대기 (재생 스레드가 실패로 끝나면 g_player_done 으로 탈출)
    while (!g_playing && !g_stop && !g_player_done)
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    if (!g_playing) {
        fprintf(stderr, "[main] 재생 시작 실패 — 정리 후 종료\n");
        g_stop = true;
        if (player.joinable()) player.join();
        led.set(0.f);
        led.close();
        motor.close(args.motor_home);
        return 1;
    }

    // ===== 동기 구동 루프 (50 Hz) =====
    const auto control_period = std::chrono::milliseconds(20);
    const double sync_off = args.sync_ms / 1000.0;
    auto next_tick = std::chrono::steady_clock::now();
    while (!g_stop) {
        double t = std::chrono::duration<double>(std::chrono::steady_clock::now() - g_play_start)
                       .count();
        if (t >= tl.duration() && !g_playing) break;

        double tt = t - sync_off;
        long idx = static_cast<long>(std::lround(tt * tl.fps));
        if (idx < 0) idx = 0;
        if (idx >= static_cast<long>(tl.size())) idx = static_cast<long>(tl.size()) - 1;

        led.set(tl.led[idx]);
        int32_t ticks = args.motor_home + static_cast<int32_t>(tl.motor[idx] * args.motor_amp);
        motor.moveTo(ticks);

        next_tick += control_period;
        std::this_thread::sleep_until(next_tick);
    }

    fprintf(stderr, "[main] 정리 중...\n");
    g_stop = true;
    if (player.joinable()) player.join();
    led.set(0.f);
    led.close();
    motor.close(args.motor_home);
    fprintf(stderr, "[main] 종료\n");
    return 0;
}
