import os
import pandas as pd
import shutil

# 1. 경로 설정 (제시해주신 경로 기준)
base_path = r'G:\stetho_ai\archive (1)'
diag_path = os.path.join(base_path, 'Respiratory_Sound_Database', 'Respiratory_Sound_Database', 'patient_diagnosis.csv')
demo_path = os.path.join(base_path, 'demographic_info.txt')
audio_dir = os.path.join(base_path, 'Respiratory_Sound_Database', 'Respiratory_Sound_Database', 'audio_and_txt_files')
target_base_dir = r'G:\stetho_ai\LUNG SOUND'

# 2. 데이터 불러오기
# demographic_info는 공백 구분자로 되어 있는 경우가 많습니다.
demo_df = pd.read_csv(demo_path, sep='\s+', header=None, names=['ID', 'Age', 'Sex', 'BMI', 'Weight', 'Height'])
# patient_diagnosis는 A열(ID), B열(질병명) 구조
diag_df = pd.read_csv(diag_path, header=None, names=['ID', 'Diagnosis'])

# 3. 데이터 병합 (ID 기준)
info_df = pd.merge(demo_df[['ID', 'Age']], diag_df, on='ID')

# 4. 파일 정리 루프
if not os.path.exists(target_base_dir):
    os.makedirs(target_base_dir)

for index, row in info_df.iterrows():
    patient_id = str(int(row['ID']))
    age = row['Age']
    diagnosis = row['Diagnosis'].strip()

    # 성인/아동 구분 (18세 기준)
    age_group = 'Adult' if age >= 18 else 'Child'

    # 대상 폴더 경로 생성: G:\stetho_ai\LUNG SOUND\Adult\COPD 형식
    target_dir = os.path.join(target_base_dir, age_group, diagnosis)

    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    # 해당 환자 ID로 시작하는 모든 파일(wav, txt) 찾아서 복사
    for filename in os.listdir(audio_dir):
        if filename.startswith(patient_id + '_'):
            source_file = os.path.join(audio_dir, filename)
            target_file = os.path.join(target_dir, filename)

            # 파일 복사 (이미 있으면 건너뜀)
            if not os.path.exists(target_file):
                shutil.copy2(source_file, target_file)

print(f"정리가 완료되었습니다! 결과 확인: {target_base_dir}")