# 서버 이전 실행 기록

2026-10-09. 사용자가 현재 RunPod 임시 디스크 사용을 승인하여 이전을 시작했다.
대상은 `root@202.181.159.219:13980`, `/workspace/komap`, L40S 1개다.
영구 볼륨 생성 대기는 취소했다. 기존 서버 파일은 삭제하지 않는다.

## 범위와 검증

- 최종 원본 목록: 5,347개, 123,784,030,984바이트. 모든 SHA256은 `transfer-manifest.json`에 기록했다.
- 자체 학습 가중치·체크포인트·분류기, 코드·Git 기록, 설정·분할·결과·보고서가 대상이다.
- 원본 데이터, 공개 사전학습 모델 캐시, OOF npy/npz, 가상환경은 복사하지 않는다.
- 데이터 재취득을 위한 2,528개 파일의 경로·크기·해시는 `dataset-file-identities.json`에 있다.
- 압축하지 않는다. HTTPS는 base64 전송 형식을 사용하며 대상에는 원래 바이트를 저장한다.
- 전체 SHA256, uv 환경 검사, 대표 체크포인트 4개의 추론 비교가 모두 통과해야 완료로 기록한다.

## 환경

`uv pip freeze`로 기존 67개 패키지를 기록하고 해시가 포함된 lock으로 복원했다.
Python 3.12.3, torch 2.14.1+cu126, torchvision 0.29.1+cu126을 그대로 사용한다.
`uv pip check`와 GPU BF16 forward 검사까지 통과했다.
대상 `backup_metadata/uv_restore_verified.json`에 결과가 있다.

## 전송

서버 간 양방향 SSH가 타임아웃되어 먼저 PC의 SSH 중계로 코드·기록을 옮겼다.
이 경로가 느려 가중치는 기존 서버에서 RunPod의 인증된 HTTPS API로 직접 전송한다.
16MiB 단위, 8개 파일 병렬 업로드이며 PC에 가중치를 저장하지 않는다.
기록 4,327개와 먼저 옮긴 가중치 7개는 SHA256 대조 후 재전송에서 제외했다.
PC 중계 프로세스와 전송 전용 SSH 키는 정리했다.
현재 업로드와 종료 후 검증은 두 서버에서 분리된 프로세스로 실행되어 PC 연결에 의존하지 않는다.

원본 서버 관리 폴더는 `/home/j-j15a106/.cache/komap_migration_20261009_runpod`다.

- `https_state.json`: 현재 단계·완료 파일·바이트·평균 속도.
- `https_upload.log`: 진행 및 파일 전체 재시도 기록.
- `migration_verified.json`: 최종 대상 검증을 통과한 경우에만 생성한다.
- `jupyter_auth.json`: 기존 대상 인증을 사용하는 임시 비밀 파일. 권한 0600이며 복사 대상에서 제외하고 완료 시 삭제한다.
- 이전 `state.json`·`transfer.log`는 중계 경로 기록이다. 현재 상태는 `https_state.json`이다.

대상 `backup_metadata` 폴더의 기록:

- `transfer_manifest.json`: 고정한 원본 목록.
- `https_transfer_complete.json`: 업로드 종료 알림이며 검증 완료를 뜻하지 않는다.
- `https_destination_verification.log`: 전체 SHA256 및 모델 검사 로그.
- `checkpoint_initial_verified.json`: 대표 모델의 초기 이전 검사 결과.
- `checkpoint_inference_verified.json`: 전체 전송 이후 대표 모델 비교 결과.
- `transfer_verified.json`: 전체 SHA256 통과 결과.
- `https_migration_result.json`: 최종 완료 또는 실패 결과.

대표 비교 모델은 2배 MiT, 2배 UNet++, 4배 MiT seed 0·1의 fold 0 마지막 체크포인트다.
128×128 인공 입력으로 strict state_dict 로딩, 유한 출력, 원본 추론과의 일치를 검사했다.
실제 데이터 점수 재현과 추가 학습은 데이터 재취득 후 별도 작업이다.

고정 원본 목록의 파일은 이전 도중 수정하지 않는다.
새 관리 코드와 실행 기록은 대상 `backup_metadata/migration_execution_20261009`에 별도로 보관한다.

공식 참고: [uv 잠금과 설치](https://docs.astral.sh/uv/pip/compile/),
[RunPod HTTPS 프록시](https://github.com/runpod/docs/blob/main/integrations/overview.mdx),
[Jupyter 대용량 업로드](https://jupyter-server.readthedocs.io/en/latest/api/jupyter_server.services.contents.html).
