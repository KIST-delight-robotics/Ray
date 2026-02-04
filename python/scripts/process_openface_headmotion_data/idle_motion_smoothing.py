#!/usr/bin/env python3
"""
OpenFace에서 추출한 헤드모션 데이터를 리샘플링하고 필터링하는 스크립트.

사용법:
    python idle_motion_smoothing.py <input_csv_file>
    
예시:
    python idle_motion_smoothing.py /path/to/headmotion_1.csv
    
출력:
    - <input_basename>_resampled_25fps.csv : 전체 리샘플링/필터링 데이터
    - <input_basename>_rpy_only.csv : roll, pitch, yaw만 추출 (헤더 없음)
    - <input_basename>_full.png : 전체 플롯
    - <input_basename>_zoomed.png : 동작 큰 구간 확대 플롯
"""

import sys
import os
import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt


def process_headmotion(input_file: str):
    """헤드모션 데이터를 처리하는 메인 함수"""
    
    # 입력 파일 경로 파싱
    input_dir = os.path.dirname(input_file)
    input_basename = os.path.splitext(os.path.basename(input_file))[0]
    
    # 출력 디렉토리 (입력 파일과 같은 디렉토리)
    output_dir = input_dir if input_dir else '.'
    
    print(f"입력 파일: {input_file}")
    print(f"출력 디렉토리: {output_dir}")
    print(f"출력 파일 prefix: {input_basename}")
    print("-" * 50)
    
    # 원본 데이터 읽기
    df = pd.read_csv(input_file)
    
    # 컬럼명 공백 제거
    df.columns = df.columns.str.strip()
    
    # 원본 FPS 확인
    original_fps = 1 / (df['timestamp'].iloc[1] - df['timestamp'].iloc[0])
    print(f"원본 FPS: {original_fps:.2f}")
    
    # 타겟 재생 주기
    target_period_ms = 40  # 40ms
    target_fps = 1000 / target_period_ms  # 25 fps
    print(f"타겟 FPS: {target_fps:.2f}")
    
    # 새로운 타임스탬프 생성 (40ms 간격)
    original_time = df['timestamp'].values
    new_time = np.arange(original_time[0], original_time[-1], target_period_ms / 1000)
    
    # 각 채널별로 보간 (리샘플링)
    resampled_data = {'timestamp': new_time}
    for col in ['pose_Rx', 'pose_Ry', 'pose_Rz']:
        # 보간 함수 생성 (cubic spline)
        f = interp1d(original_time, df[col].values, kind='cubic', fill_value='extrapolate')
        resampled_data[col] = f(new_time)
    
    df_resampled = pd.DataFrame(resampled_data)
    
    # 리샘플링 후 필터링 (25fps 기준 파라미터)
    window_sec = 0.5  # 0.5초 윈도우
    window_length = int(target_fps * window_sec)
    if window_length % 2 == 0:
        window_length += 1  # 홀수로 만들기
    
    print(f"필터 window_length: {window_length} (약 {window_length/target_fps:.2f}초)")
    
    # Savgol 필터 적용
    df_filtered = df_resampled.copy()
    for col in ['pose_Rx', 'pose_Ry', 'pose_Rz']:
        df_filtered[f'smoothed_{col}'] = savgol_filter(
            df_resampled[col], 
            window_length=window_length, 
            polyorder=3
        )
    
    print(f"원본 프레임 수: {len(df)}")
    print(f"리샘플링 후 프레임 수: {len(df_resampled)}")
    
    # === 동작이 큰 구간 찾기 ===
    window_detect = 50  # 탐지용 윈도우
    df_resampled['motion_magnitude'] = 0
    for col in ['pose_Rx', 'pose_Ry', 'pose_Rz']:
        df_resampled['motion_magnitude'] += df_resampled[col].rolling(window=window_detect, center=True).std().fillna(0)
    
    # 가장 동작이 큰 구간 찾기
    max_motion_idx = df_resampled['motion_magnitude'].idxmax()
    max_motion_time = df_resampled.loc[max_motion_idx, 'timestamp']
    zoom_start = max(0, max_motion_time - 3)
    zoom_end = min(df_resampled['timestamp'].iloc[-1], max_motion_time + 3)
    
    print(f"\n동작이 가장 큰 시간: {max_motion_time:.2f}초")
    print(f"확대 구간: {zoom_start:.2f}초 ~ {zoom_end:.2f}초")
    
    # === 시각화 공통 설정 ===
    # pose_Rz -> Roll, pose_Rx -> Pitch, pose_Ry -> Yaw
    labels = {'pose_Rx': 'Pitch (Rx)', 'pose_Ry': 'Yaw (Ry)', 'pose_Rz': 'Roll (Rz)'}
    colors = {'original': '#3498db', 'resampled': '#e74c3c', 'filtered': '#2ecc71'}
    
    def plot_comparison(axes, xlim=None):
        for i, col in enumerate(['pose_Rx', 'pose_Ry', 'pose_Rz']):
            ax = axes[i]
            
            ax.plot(df['timestamp'], df[col], 
                    color=colors['original'], alpha=0.5, linewidth=0.8,
                    label=f'Original ({original_fps:.0f}fps)')
            
            ax.plot(df_resampled['timestamp'], df_resampled[col], 
                    color=colors['resampled'], alpha=0.7, linewidth=1.0, linestyle='--',
                    label=f'Resampled ({target_fps:.0f}fps)')
            
            ax.plot(df_filtered['timestamp'], df_filtered[f'smoothed_{col}'], 
                    color=colors['filtered'], alpha=0.9, linewidth=1.5,
                    label=f'Filtered (Savgol)')
            
            ax.set_ylabel(labels[col], fontsize=11)
            ax.legend(loc='upper right')
            ax.grid(True, alpha=0.3)
            
            if xlim:
                ax.set_xlim(xlim)
        
        axes[-1].set_xlabel('Time (seconds)', fontsize=11)
    
    # === 1. 전체 플롯 ===
    fig1, axes1 = plt.subplots(3, 1, figsize=(16, 10))
    plot_comparison(axes1, xlim=None)
    fig1.suptitle('Head Motion RPY: Full Data', fontsize=14, fontweight='bold')
    plt.tight_layout()
    full_plot_path = os.path.join(output_dir, f'{input_basename}_full.png')
    plt.savefig(full_plot_path, dpi=150, bbox_inches='tight')
    print(f"\n전체 플롯 저장: {full_plot_path}")
    
    # === 2. 동작이 큰 구간 확대 플롯 ===
    fig2, axes2 = plt.subplots(3, 1, figsize=(14, 10))
    plot_comparison(axes2, xlim=(zoom_start, zoom_end))
    fig2.suptitle(f'Head Motion RPY: High Motion Zone ({zoom_start:.1f}s ~ {zoom_end:.1f}s)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    zoomed_plot_path = os.path.join(output_dir, f'{input_basename}_zoomed.png')
    plt.savefig(zoomed_plot_path, dpi=150, bbox_inches='tight')
    print(f"확대 플롯 저장: {zoomed_plot_path}")
    
    # === 데이터 저장 ===
    # 전체 리샘플링/필터링 데이터
    full_csv_path = os.path.join(output_dir, f'{input_basename}_resampled_25fps.csv')
    df_filtered.to_csv(full_csv_path, index=False)
    print(f"전체 데이터 저장: {full_csv_path}")
    
    # 결과 열 3개만 추출 (rz=roll, rx=pitch, ry=yaw 순서), 헤더 없이 저장
    df_result_only = df_filtered[['smoothed_pose_Rz', 'smoothed_pose_Rx', 'smoothed_pose_Ry']]
    rpy_only_path = os.path.join(output_dir, f'{input_basename}_rpy_only.csv')
    df_result_only.to_csv(rpy_only_path, index=False, header=False)
    print(f"RPY 전용 데이터 저장: {rpy_only_path} ({len(df_result_only)} rows)")
    
    print("\n처리 완료!")
    return rpy_only_path


def main():
    if len(sys.argv) < 2:
        print("사용법: python idle_motion_smoothing.py <input_csv_file>")
        print("예시: python idle_motion_smoothing.py /path/to/headmotion_1.csv")
        sys.exit(1)
    
    input_file = sys.argv[1]
    
    if not os.path.exists(input_file):
        print(f"오류: 파일을 찾을 수 없습니다 - {input_file}")
        sys.exit(1)
    
    process_headmotion(input_file)


if __name__ == '__main__':
    main()