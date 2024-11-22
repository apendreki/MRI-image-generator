# ===========================
# Import 및 설정
# ===========================
import os
import sys
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import nibabel as nib
import json
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import re
import logging
from nibabel.orientations import io_orientation
import time
import multiprocessing
from torch.amp import autocast, GradScaler
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

# ===========================
# 로그 설정
# ===========================
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')

# ===========================
# 난수 시드 설정
# ===========================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # multi-GPU 사용 시

    # 성능 향상을 위해 cudnn.benchmark를 True로 설정하고, deterministic을 False로 설정합니다.
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

SEED = 42
set_seed(SEED)

# ===========================
# 설정값
# ===========================
DATASET_ROOT = "./dataset/MICCAI-BraTS2024-MET-Challenge-Training_1"  # 학습 데이터셋의 루트 디렉토리
RESIZE_DIM = 512  # 입력 이미지의 크기 (512x512)로 원본 해상도 사용

# 실행 설정 리스트
RUN_SETTINGS_LIST = [
    # 첫 번째 설정: FILM 모델로 학습 후 테스트
    {
        'mode': 'train',
        'model_type': 'film',
        'settings': {
            'modality': 't1n',
            'num_epochs': 30,
            'learning_rate': 0.01,
            'save_every': 5,
            'batch_size': 2,  # 메모리 이슈를 피하기 위해 배치 크기를 작게 설정
            'use_tqdm': True,
            'model_path': None,
            'validation_split': 0.1,
            'window_size': 5,
            'augment_data': True,
            'loss_type': 'L1',
            'num_workers': 4,  # 멀티프로세싱 비활성화
            'pin_memory': True,
            'prefetch_factor': 2,
            'use_amp': True,
            'patch_size': 256,
            'use_patch_training': True,
            'scheduler_type': 'StepLR',
            'scheduler_params': {'step_size': 10, 'gamma': 0.1},
        }
    },
    {
        'mode': 'test',
        'model_type': 'film',
        'settings': {
            'start_image_path': './extracted_images/slice_0110.png',
            'start_z_mm': 110.0,
            'end_image_path': './extracted_images/slice_0120.png',
            'end_z_mm': 120.0,
            'num_frames': 5,
            'original_nifti_path': None,
            'model_path': './models/film/film_L1_epoch_10.pth',
            'save_path': './test_output/film'
        }
    },
    # 두 번째 설정: DiffMorpher 모델로 학습 후 테스트
    {
        'mode': 'train',
        'model_type': 'diffmorpher',
        'settings': {
            'modality': 't1n',
            'num_epochs': 10,
            'learning_rate': 0.0001,
            'save_every': 5,
            'batch_size': 2,  # 메모리 이슈를 피하기 위해 배치 크기를 작게 설정
            'use_tqdm': True,
            'model_path': None,
            'validation_split': 0.1,
            'window_size': 5,
            'augment_data': True,
            'loss_type': 'L1',
            'num_workers': 4,  # 멀티프로세싱 비활성화
            'pin_memory': True,
            'prefetch_factor': 2,
            'use_amp': True,
            'patch_size': 256,
            'use_patch_training': True,
            'scheduler_type': 'StepLR',
            'scheduler_params': {'step_size': 5, 'gamma': 0.1},
        }
    },
    {
        'mode': 'test',
        'model_type': 'diffmorpher',
        'settings': {
            'start_image_path': './extracted_images/slice_0110.png',
            'start_z_mm': 110.0,
            'end_image_path': './extracted_images/slice_0120.png',
            'end_z_mm': 120.0,
            'num_frames': 5,
            'original_nifti_path': None,
            'model_path': './models/diffmorpher/diffmorpher_L1_epoch_10.pth',
            'save_path': './test_output/diffmorpher'
        }
    },
    # 추가 설정을 여기에 추가할 수 있습니다.
]

PREPROCESS_SETTINGS = {
    'preprocessed_data_dir': './preprocessed_data',
    'num_workers': 0  # 멀티프로세싱 비활성화
}

EXTRACT_SETTINGS = {
    'nifti_path': './dataset/MICCAI-BraTS2021-MET-Challenge-Training_1/BraTS-MET-00001-000/BraTS-MET-00001-000-t1n.nii.gz',
    'output_dir': './extracted_images',
    'positions': [110, 120],
    'position_unit': 'slice',
    'modality': 't1n'
}

# ===========================
# 데이터셋 클래스 정의
# ===========================
class BraTSDataset(Dataset):
    """
    BraTS 데이터셋을 로드하는 커스텀 Dataset 클래스입니다.
    """
    def __init__(self, dataset_root, transform=None, modality='t1n', mode='mmap', cache_limit=10, window_size=5, augment_data=False, preprocessed=False, patch_size=None, use_patch_training=False):
        """
        Args:
            dataset_root (str): 데이터셋 루트 디렉토리 경로
            transform (callable, optional): 전처리를 위한 함수
            modality (str): 사용할 MRI 모달리티 (예: 't1n', 't1c', 't2w', 't2f', 'seg')
            mode (str): 데이터 로딩 모드 ('mmap', 'cache' 등)
            cache_limit (int): 캐시할 슬라이스의 최대 개수
            window_size (int): 한 번에 로드할 슬라이스의 수
            augment_data (bool): 데이터 증강 여부
            preprocessed (bool): 전처리된 데이터 사용 여부
            patch_size (int): 패치 크기
            use_patch_training (bool): 패치 기반 학습 사용 여부
        """
        self.dataset_root = dataset_root
        self.transform = transform
        self.modality = modality
        self.mode = mode
        self.cache_limit = cache_limit
        self.window_size = window_size
        self.augment_data = augment_data
        self.preprocessed = preprocessed
        self.patch_size = patch_size
        self.use_patch_training = use_patch_training

        # NIfTI 이미지 캐싱을 위한 딕셔너리 (멀티프로세싱 호환성을 위해 제거)
        # self.nifti_cache = {}

        # 데이터셋 구조 검증
        self.validate_dataset_structure()

        # 데이터 준비
        self.prepare_slices()

    def validate_dataset_structure(self):
        """
        데이터셋의 구조와 파일 형식을 검증합니다.
        """
        required_modalities = [self.modality]
        all_valid = True
        patient_dirs = sorted(os.listdir(self.dataset_root))
        for patient_dir in patient_dirs:
            patient_path = os.path.join(self.dataset_root, patient_dir)
            if os.path.isdir(patient_path):
                missing_modalities = []
                for modality in required_modalities:
                    pattern = re.compile(rf'{re.escape(patient_dir)}.*-{modality}\.nii\.gz$', re.IGNORECASE)
                    matches = [f for f in os.listdir(patient_path) if pattern.match(f)]
                    if not matches:
                        missing_modalities.append(modality)
                        all_valid = False
                        logging.warning(f"환자 {patient_dir}에서 모달리티 '{modality}' 파일을 찾을 수 없습니다.")
                    else:
                        # 각 모달리티 파일의 형식 검증
                        file = matches[0]
                        file_path = os.path.join(patient_path, file)
                        try:
                            nib.load(file_path)  # 파일이 올바른 형식인지 확인
                        except Exception as e:
                            logging.error(f"환자 {patient_dir}의 파일 {file} 로드 오류: {e}")
                            all_valid = False
        if not all_valid:
            raise ValueError("데이터셋 검증 실패: 누락된 모달리티 파일 또는 파일 형식 오류가 발견되었습니다.")
        logging.info("데이터셋 구조 검증 성공.")

    def prepare_slices(self):
        """
        데이터셋 디렉토리를 탐색하여 사용할 슬라이스 목록을 준비합니다.
        """
        self.slices = []
        patient_dirs = sorted(os.listdir(self.dataset_root))
        for patient_dir in patient_dirs:
            patient_path = os.path.join(self.dataset_root, patient_dir)
            if os.path.isdir(patient_path):
                # 지정된 모달리티에 해당하는 파일 찾기
                modality_pattern = re.compile(rf'{re.escape(patient_dir)}.*-{self.modality}\.nii\.gz$', re.IGNORECASE)
                modality_file = [f for f in os.listdir(patient_path) if modality_pattern.match(f)]
                if modality_file:
                    img_path = os.path.join(patient_path, modality_file[0])
                    try:
                        # nifti_cache 제거로 인한 수정
                        nii_img = nib.load(img_path, mmap=False)  # 메모리 맵핑 사용 안 함
                        img_data = nii_img.get_fdata()
                        img_shape = img_data.shape
                        if len(img_shape) >= 3 and img_shape[2] >= self.window_size + 2:
                            for i in range(img_shape[2] - self.window_size - 1):
                                self.slices.append((img_path, i, img_shape[2]))
                    except Exception as e:
                        logging.error(f"{img_path} 로드 중 오류 발생: {e}")
                else:
                    logging.warning(f"{patient_path}에서 모달리티 '{self.modality}' 파일을 찾을 수 없습니다.")

    def load_slice(self, img_path, slice_index):
        """
        지정된 슬라이스 인덱스에서 이미지를 로드합니다.
        """
        # nifti_cache 제거로 인한 수정
        nii_img = nib.load(img_path, mmap=False)
        img_data = nii_img.get_fdata()
        affine = nii_img.affine

        # 슬라이스 추출
        img_slice = img_data[:, :, slice_index]
        # 방향 조정
        orientation = io_orientation(affine)
        if orientation[0, 0] == -1:
            img_slice = np.flip(img_slice, axis=0)
        if orientation[1, 1] == -1:
            img_slice = np.flip(img_slice, axis=1)
        # 이미지 정규화
        img_slice = (img_slice - np.min(img_slice)) / (np.max(img_slice) - np.min(img_slice) + 1e-5)
        img_slice = (img_slice * 255).astype(np.uint8)
        img_data = Image.fromarray(img_slice)

        return img_data

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, idx):
        """
        인덱스에 해당하는 데이터를 반환합니다.

        Returns:
            tuple: (start_frame, mid_frames_tensor, end_frame, z_pos)
        """
        img_path, slice_index, total_slices = self.slices[idx]
        frames = []
        for i in range(self.window_size + 2):
            current_slice = slice_index + i
            frame = self.load_slice(img_path, current_slice)
            frames.append(frame)

        # 데이터 증강 (필요한 경우)
        if self.augment_data:
            augment_transform = []
            if random.random() > 0.5:
                augment_transform.append(transforms.RandomHorizontalFlip(1.0))
            if random.random() > 0.5:
                augment_transform.append(transforms.RandomVerticalFlip(1.0))
            if augment_transform:
                augment = transforms.Compose(augment_transform)
                frames = [augment(frame) for frame in frames]

        # Transform 적용
        if self.transform:
            frames = [self.transform(frame) for frame in frames]
        else:
            frames = [transforms.ToTensor()(frame) for frame in frames]

        # 시작 프레임, 중간 프레임, 종료 프레임 분리
        start_frame = frames[0]
        end_frame = frames[-1]
        mid_frames = frames[1:-1]

        # 패치 기반 학습 적용
        if self.use_patch_training and self.patch_size:
            # 랜덤 패치 추출
            i, j, h, w = transforms.RandomCrop.get_params(
                start_frame, output_size=(self.patch_size, self.patch_size)
            )
            start_frame = start_frame[:, i:i+h, j:j+w]
            end_frame = end_frame[:, i:i+h, j:j+w]
            mid_frames = [frame[:, i:i+h, j:j+w] for frame in mid_frames]

        # 텐서로 변환
        mid_frames = torch.stack(mid_frames)  # [window_size, C, H, W]
        z_positions = torch.linspace(0, 1, steps=self.window_size + 2)[1:-1]  # 중간 프레임들의 위치 정보

        return start_frame, mid_frames, end_frame, z_positions

# ===========================
# 모델 클래스 정의
# ===========================
class ResidualBlock(nn.Module):
    """
    Residual Block을 구현한 클래스.

    Args:
        in_channels (int): 입력 채널 수.
    """
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.norm1 = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = nn.BatchNorm2d(in_channels)

    def forward(self, x):
        """
        Residual 블록의 순전파 함수.

        Args:
            x (torch.Tensor): 입력 텐서.

        Returns:
            torch.Tensor: 출력 텐서.
        """
        residual = x
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.norm2(out)
        return out + residual

class BidirectionalFlowEstimator(nn.Module):
    """
    양방향 움직임 예측기입니다.
    """
    def __init__(self, in_channels):
        super(BidirectionalFlowEstimator, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels * 2, 64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 4, kernel_size=4, stride=2, padding=1)  # 2 for forward flow, 2 for backward flow
        )

    def forward(self, start, end):
        x = torch.cat([start, end], dim=1)  # [B, 2*C, H, W]
        x = self.encoder(x)
        flow = self.decoder(x)  # [B, 4, H', W']
        flow_f = flow[:, :2, :, :]  # Forward flow
        flow_b = flow[:, 2:, :, :]  # Backward flow
        return flow_f, flow_b

class Warp(nn.Module):
    """
    워핑 모듈입니다.
    """
    def __init__(self):
        super(Warp, self).__init__()

    def forward(self, x, flow):
        B, C, H, W = x.size()
        # 'ij' 인덱싱을 명시적으로 지정하여 torch.meshgrid 호출
        grid_y, grid_x = torch.meshgrid(
            torch.arange(0, H, device=x.device),
            torch.arange(0, W, device=x.device),
            indexing='ij'
        )
        grid = torch.stack((grid_x, grid_y), 2).float()  # [H, W, 2]
        grid = grid.unsqueeze(0).repeat(B, 1, 1, 1)  # [B, H, W, 2]

        # Normalize grid to [-1, 1]
        grid[:, :, :, 0] = 2.0 * grid[:, :, :, 0] / (W - 1) - 1.0
        grid[:, :, :, 1] = 2.0 * grid[:, :, :, 1] / (H - 1) - 1.0

        # Apply flow (assuming flow is in pixels)
        flow_norm = torch.zeros_like(flow)
        flow_norm[:, 0, :, :] = 2.0 * flow[:, 0, :, :] / (W - 1)
        flow_norm[:, 1, :, :] = 2.0 * flow[:, 1, :, :] / (H - 1)
        flow_norm = flow_norm.permute(0, 2, 3, 1)
        grid = grid + flow_norm

        output = nn.functional.grid_sample(x, grid, align_corners=True, mode='bilinear', padding_mode='border')
        return output

class SynthesisNetwork(nn.Module):
    """
    합성 네트워크입니다.
    """
    def __init__(self, in_channels):
        super(SynthesisNetwork, self).__init__()
        self.conv1 = nn.Conv2d(in_channels * 2, 64, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(64) for _ in range(5)]
        )
        self.conv2 = nn.Conv2d(64, 1, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, warped_start, warped_end):
        x = torch.cat([warped_start, warped_end], dim=1)  # [B, 2*C, H, W]
        x = self.relu1(self.conv1(x))
        x = self.res_blocks(x)
        x = self.conv2(x)
        x = self.sigmoid(x)
        return x

class FILMModel(nn.Module):
    """
    FILM 모델입니다.
    """
    def __init__(self):
        super(FILMModel, self).__init__()
        self.flow_estimator = BidirectionalFlowEstimator(in_channels=1)
        self.warp = Warp()
        self.synthesis = SynthesisNetwork(in_channels=1)

    def forward(self, start, end, t):
        """
        Args:
            start (torch.Tensor): 시작 프레임 [B, 1, H, W]
            end (torch.Tensor): 종료 프레임 [B, 1, H, W]
            t (torch.Tensor): 중간 위치 [B, 1, 1, 1]

        Returns:
            torch.Tensor: 생성된 중간 프레임 [B, 1, H, W]
        """
        flow_f, flow_b = self.flow_estimator(start, end)

        # t 크기 조정
        t = t.view(-1, 1, 1, 1)
        flow_t = t * flow_f - (1 - t) * flow_b

        warped_start = self.warp(start, flow_t)
        warped_end = self.warp(end, -flow_t)

        output = self.synthesis(warped_start, warped_end)
        return output

class Encoder(nn.Module):
    """
    인코더 네트워크입니다.
    """
    def __init__(self, latent_dim=256):
        super(Encoder, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, latent_dim, kernel_size=4, stride=2, padding=1)
        )

    def forward(self, x):
        return self.model(x)

class Decoder(nn.Module):
    """
    디코더 네트워크입니다.
    """
    def __init__(self, latent_dim=256):
        super(Decoder, self).__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def forward(self, z):
        return self.model(z)

class DiffMorpher(nn.Module):
    """
    DiffMorpher 모델입니다.
    """
    def __init__(self, latent_dim=256):
        super(DiffMorpher, self).__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)

    def forward(self, start, end, t):
        """
        Args:
            start (torch.Tensor): 시작 프레임 [B, 1, H, W]
            end (torch.Tensor): 종료 프레임 [B, 1, H, W]
            t (torch.Tensor): 중간 위치 [B, 1, 1, 1]

        Returns:
            torch.Tensor: 생성된 중간 프레임 [B, 1, H, W]
        """
        z_start = self.encoder(start)
        z_end = self.encoder(end)
        z_t = z_start + (z_end - z_start) * t

        # 노이즈 추가 (확률적 보간)
        noise = torch.randn_like(z_t) * 0.1
        z_t = z_t + noise

        output = self.decoder(z_t)
        return output

# ===========================
# 학습 및 테스트 함수 정의
# ===========================
def train_model(generator, train_loader, val_loader=None, num_epochs=10,
               learning_rate=0.0001, save_every=5, use_tqdm=True,
               model_dir="models", start_epoch=0, loss_type='L1',
               use_amp=False, device=None, model_name='model',
               scheduler_type='StepLR', scheduler_params=None):
    """
    학습 함수 정의

    Args:
        generator (nn.Module): 생성자 모델
        train_loader (DataLoader): 학습 데이터 로더
        val_loader (DataLoader, optional): 검증 데이터 로더
        num_epochs (int): 학습할 총 에폭 수
        learning_rate (float): 학습률
        save_every (int): 몇 에폭마다 모델을 저장할지 설정
        use_tqdm (bool): tqdm 프로그레스 바 사용 여부
        model_dir (str): 모델을 저장할 디렉토리 경로
        start_epoch (int): 학습을 시작할 에폭 번호
        loss_type (str): 'L1' 또는 'L2' 손실 함수 선택
        use_amp (bool): 혼합 정밀도 학습 사용 여부
        device (torch.device): 장치 설정
        model_name (str): 모델 이름 (저장 시 사용)
        scheduler_type (str): 스케줄러 타입 ('StepLR', 'ReduceLROnPlateau' 등)
        scheduler_params (dict): 스케줄러 파라미터
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = generator.to(device)

    # 다중 GPU 사용을 위해 DataParallel 적용
    if torch.cuda.device_count() > 1:
        logging.info(f"{torch.cuda.device_count()}개의 GPU를 사용합니다.")
        generator = nn.DataParallel(generator)
    else:
        logging.info("1개의 GPU를 사용합니다.")

    g_optimizer = optim.Adam(generator.parameters(), lr=learning_rate)

    # 손실 함수 정의
    if loss_type == 'L1':
        criterion = nn.L1Loss()
    elif loss_type == 'L2':
        criterion = nn.MSELoss()
    else:
        raise ValueError("Invalid loss_type. Choose 'L1' or 'L2'.")

    # 스케줄러 설정
    if scheduler_type == 'StepLR':
        scheduler = optim.lr_scheduler.StepLR(g_optimizer, **scheduler_params)
    elif scheduler_type == 'ReduceLROnPlateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(g_optimizer, **scheduler_params)
    else:
        raise ValueError("Invalid scheduler_type. Choose 'StepLR' or 'ReduceLROnPlateau'.")

    os.makedirs(model_dir, exist_ok=True)

    # TensorBoard 설정
    writer = SummaryWriter(log_dir=os.path.join(model_dir, model_name))

    # 학습 로그 초기화 또는 로드
    log_file_path = os.path.join(model_dir, model_name, "training_log.json")
    if os.path.exists(log_file_path):
        with open(log_file_path, "r") as log_file:
            log_data = json.load(log_file)
    else:
        log_data = {"losses": []}

    # 혼합 정밀도 학습 스케일러 초기화
    scaler = GradScaler(enabled=use_amp)

    for epoch in range(start_epoch, num_epochs):
        generator.train()
        running_loss = 0.0  # 에포크 손실 누적 변수

        if use_tqdm:
            progress_bar = tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{num_epochs}")

        start_time = time.time()

        for batch_idx, data in enumerate(train_loader):
            start_frame, mid_frames, end_frame, z_pos = data
            start_frame = start_frame.to(device, non_blocking=True)
            end_frame = end_frame.to(device, non_blocking=True)
            mid_frames = mid_frames.to(device, non_blocking=True)
            z_pos = z_pos.to(device, non_blocking=True)  # [window_size]

            batch_size, num_mid_frames, C, H, W = mid_frames.shape
            batch_loss = 0.0  # 배치 손실 초기화

            for i in range(num_mid_frames):
                mid_frame = mid_frames[:, i, :, :, :]  # 실제 중간 프레임 [batch_size, C, H, W]
                current_z_pos = z_pos[i].view(1, 1, 1, 1).repeat(batch_size, 1, 1, 1).to(device)  # [batch_size, 1, 1, 1]

                with autocast(enabled=use_amp):
                    interpolated_frame = generator(start_frame, end_frame, current_z_pos)
                    loss = criterion(interpolated_frame, mid_frame)

                batch_loss += loss.item()

                # 역전파 및 최적화
                g_optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(g_optimizer)
                scaler.update()

            running_loss += batch_loss / num_mid_frames

            if use_tqdm:
                progress_bar.update(1)
                progress_bar.set_postfix({"Loss": f"{batch_loss / num_mid_frames:.4f}"})

            # TensorBoard에 손실 값 기록
            writer.add_scalar('Loss', batch_loss / num_mid_frames, epoch * len(train_loader) + batch_idx)

        end_time = time.time()
        epoch_time = end_time - start_time

        if use_tqdm:
            progress_bar.close()

        # 평균 손실 계산
        avg_loss = running_loss / len(train_loader)
        logging.info(f"Epoch [{epoch+1}/{num_epochs}] Completed. Average Loss: {avg_loss:.4f}. Time: {epoch_time:.2f}s")

        # TensorBoard에 에포크별 평균 손실 기록
        writer.add_scalar('Average Loss', avg_loss, epoch)
        writer.add_scalar('Epoch Time', epoch_time, epoch)

        # 학습률 스케줄러 스텝 진행
        if scheduler_type == 'ReduceLROnPlateau':
            scheduler.step(avg_loss)
        else:
            scheduler.step()

        # 로그 데이터 저장
        log_data["losses"].append({"epoch": epoch + 1, "average_loss": avg_loss})
        with open(log_file_path, "w") as log_file:
            json.dump(log_data, log_file)

        # 검증 단계
        if val_loader is not None:
            generator.eval()
            val_loss = 0.0
            with torch.no_grad():
                for start_frame, mid_frames, end_frame, z_pos in val_loader:
                    start_frame = start_frame.to(device, non_blocking=True)
                    end_frame = end_frame.to(device, non_blocking=True)
                    mid_frames = mid_frames.to(device, non_blocking=True)
                    z_pos = z_pos.to(device, non_blocking=True)

                    batch_size, num_mid_frames, C, H, W = mid_frames.shape
                    # 검증용으로 중간 프레임의 중간 위치만 사용
                    mid_idx = num_mid_frames // 2
                    mid_frame = mid_frames[:, mid_idx, :, :, :].to(device)
                    current_z_pos = z_pos[mid_idx].view(1, 1, 1, 1).repeat(batch_size, 1, 1, 1).to(device)

                    with autocast(enabled=use_amp):
                        interpolated_frame = generator(start_frame, end_frame, current_z_pos)
                        loss = criterion(interpolated_frame, mid_frame)
                    val_loss += loss.item()
            avg_val_loss = val_loss / len(val_loader)
            logging.info(f"Validation Loss: {avg_val_loss:.4f}")
            # TensorBoard에 검증 손실 기록
            writer.add_scalar('Validation Loss', avg_val_loss, epoch)

        # 모델 저장 (체크포인트)
        if (epoch + 1) % save_every == 0:
            model_save_name = f"{model_name}_epoch_{epoch+1}.pth"
            model_path = os.path.join(model_dir, model_save_name)
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            try:
                # 모델을 CPU로 이동하여 저장 (DataParallel 사용 시 state_dict 수정)
                if isinstance(generator, nn.DataParallel):
                    generator_to_save = generator.module.cpu()
                else:
                    generator_to_save = generator.cpu()
                torch.save({
                    'epoch': epoch + 1,
                    'generator_state_dict': generator_to_save.state_dict(),
                    'g_optimizer_state_dict': g_optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'loss_type': loss_type,
                    'model_type': model_name,
                }, model_path)
                logging.info(f"모델이 저장되었습니다: {model_path}")
                generator_to_save.to(device)
                generator.to(device)  # 다시 모델을 원래 장치로 이동
            except (FileNotFoundError, OSError) as e:
                logging.error(f"모델 저장 오류 발생: {e}")

    writer.close()

def test_model(generator, start_image, end_image, start_z_mm, end_z_mm, num_frames=50,
               save_image=True, save_nifti=True, image_save_path="./test_output_images",
               nifti_save_path="./test_output_nifti", original_nifti_path=None,
               image_format='png', device=None):
    """
    테스트 함수 정의: 두 이미지 사이의 중간 프레임을 생성하고 저장합니다.

    Args:
        generator (nn.Module): 학습된 생성자 모델
        start_image (PIL.Image): 시작 이미지
        end_image (PIL.Image): 종료 이미지
        start_z_mm (float): 시작 z 위치 (밀리미터 단위)
        end_z_mm (float): 종료 z 위치 (밀리미터 단위)
        num_frames (int): 생성할 중간 프레임의 수
        save_image (bool): 중간 프레임 이미지를 저장할지 여부
        save_nifti (bool): 생성된 중간 프레임을 NIfTI 파일로 저장할지 여부
        image_save_path (str): 중간 프레임 이미지의 저장 경로
        nifti_save_path (str): 생성된 NIfTI 파일의 저장 경로
        original_nifti_path (str, optional): 원본 NIfTI 파일의 경로 (NIfTI 파일을 재구성할 때 사용)
        image_format (str): 저장할 이미지 형식 (예: 'png', 'jpg')
        device (torch.device): 장치 설정
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if save_image:
        os.makedirs(image_save_path, exist_ok=True)
    if save_nifti:
        os.makedirs(nifti_save_path, exist_ok=True)
    generator = generator.to(device).eval()

    # DataParallel로 래핑된 경우
    if isinstance(generator, nn.DataParallel):
        generator = generator.module

    preprocess = transforms.Compose([
        transforms.Resize((RESIZE_DIM, RESIZE_DIM)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    # 시작 및 끝 이미지 전처리
    start_frame = preprocess(start_image).unsqueeze(0).to(device)  # [1, C, H, W]
    end_frame = preprocess(end_image).unsqueeze(0).to(device)      # [1, C, H, W]

    # z_pos 값 생성 (0에서 1 사이를 num_frames로 분할)
    z_positions = np.linspace(0, 1, num_frames + 2)[1:-1]

    interpolated_frames = []
    psnr_values = []
    ssim_values = []
    z_data = []  # z_pos 데이터 저장용 리스트

    with torch.no_grad():
        for idx, z_pos in enumerate(z_positions):
            z_tensor = torch.tensor([z_pos], dtype=torch.float32).view(1, 1, 1, 1).to(device)  # [1, 1, 1, 1]
            with autocast(enabled=True):
                interpolated_frame = generator(start_frame, end_frame, z_tensor)  # [1, 1, H, W]

            if save_image:
                frame_image = interpolated_frame.squeeze(0).cpu()
                frame_image = (frame_image * 0.5 + 0.5).clamp(0, 1)  # Normalize to [0,1]
                frame_image_pil = transforms.ToPILImage()(frame_image)

                # 시각적 비교 이미지 생성
                comparison_image = Image.new('RGB', (RESIZE_DIM * 3, RESIZE_DIM))
                comparison_image.paste(start_image.resize((RESIZE_DIM, RESIZE_DIM)), (0, 0))
                comparison_image.paste(frame_image_pil.resize((RESIZE_DIM, RESIZE_DIM)), (RESIZE_DIM, 0))
                comparison_image.paste(end_image.resize((RESIZE_DIM, RESIZE_DIM)), (RESIZE_DIM * 2, 0))
                # 파일 이름에 z_pos 포함
                comparison_filename = f"frame_{idx:03d}_z_{z_pos:.2f}_comparison.{image_format}"
                comparison_image.save(os.path.join(image_save_path, comparison_filename))
                logging.info(f"개별 프레임 비교 저장: {comparison_filename}")
                # z_pos 데이터 저장
                z_data.append({"frame": idx, "z_pos": float(z_pos)})

            # PSNR 및 SSIM 계산 (여기서는 시작 이미지와 비교)
            interpolated_frame_np = frame_image.numpy().transpose(1, 2, 0).squeeze()
            start_frame_np = start_frame.squeeze(0).cpu().numpy().transpose(1, 2, 0).squeeze()
            current_psnr = psnr(start_frame_np, interpolated_frame_np, data_range=1)
            current_ssim = ssim(start_frame_np, interpolated_frame_np, data_range=1)
            psnr_values.append(current_psnr)
            ssim_values.append(current_ssim)

            interpolated_frames.append(frame_image.cpu())

    # 평가 지표 저장
    if len(psnr_values) > 0:
        avg_psnr = np.mean(psnr_values)
        avg_ssim = np.mean(ssim_values)
        with open(os.path.join(image_save_path, "evaluation_metrics.txt"), "w") as f:
            f.write(f"Average PSNR: {avg_psnr}\n")
            f.write(f"Average SSIM: {avg_ssim}\n")
        logging.info(f"평균 PSNR: {avg_psnr:.2f}, 평균 SSIM: {avg_ssim:.4f}")

    # z_pos 데이터 저장
    if save_image and len(z_data) > 0:
        with open(os.path.join(image_save_path, "z_positions.json"), "w") as f:
            json.dump(z_data, f, indent=4)
        logging.info(f"z_positions.json 파일이 저장되었습니다: {os.path.join(image_save_path, 'z_positions.json')}")

    if save_nifti:
        volume = torch.stack(interpolated_frames).squeeze(1).numpy()  # [num_frames, H, W]

        if original_nifti_path:
            try:
                original_nii = nib.load(original_nifti_path)
                affine = original_nii.affine
                header = original_nii.header.copy()
                header['pixdim'][3] = (end_z_mm - start_z_mm) / (num_frames + 1)
            except Exception as e:
                logging.error(f"NIfTI 파일 로드 중 오류 발생: {e}")
                affine = np.eye(4)
                header = None
        else:
            affine = np.eye(4)
            header = None

        nii_img = nib.Nifti1Image(volume, affine=affine, header=header)

        nii_save_path_full = os.path.join(nifti_save_path, "interpolated_volume.nii.gz")
        nib.save(nii_img, nii_save_path_full)
        logging.info(f"NIfTI 파일이 저장되었습니다: {nii_save_path_full}")

def preprocess_dataset(dataset_root, preprocessed_data_dir, modality='t1n', num_workers=0):
    """
    데이터셋을 전처리하여 저장합니다.

    Args:
        dataset_root (str): 원본 데이터셋 루트 디렉토리
        preprocessed_data_dir (str): 전처리된 데이터를 저장할 디렉토리
        modality (str): 사용할 MRI 모달리티
        num_workers (int): 병렬 처리를 위한 worker 수
    """
    os.makedirs(preprocessed_data_dir, exist_ok=True)
    patient_dirs = sorted(os.listdir(dataset_root))

    def process_patient(patient_dir):
        patient_path = os.path.join(dataset_root, patient_dir)
        if os.path.isdir(patient_path):
            modality_pattern = re.compile(rf'{re.escape(patient_dir)}.*-{modality}\.nii\.gz$', re.IGNORECASE)
            modality_file = [f for f in os.listdir(patient_path) if modality_pattern.match(f)]
            if modality_file:
                img_path = os.path.join(patient_path, modality_file[0])
                try:
                    nii_img = nib.load(img_path)
                    img = nii_img.get_fdata()
                    # 이미지 정규화
                    img = (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-5)
                    # 방향 조정
                    orientation = io_orientation(nii_img.affine)
                    if orientation[0, 0] == -1:
                        img = np.flip(img, axis=0)
                    if orientation[1, 1] == -1:
                        img = np.flip(img, axis=1)
                    # 슬라이스별로 저장
                    for slice_index in range(img.shape[2]):
                        img_slice = img[:, :, slice_index]
                        img_data = Image.fromarray((img_slice * 255).astype(np.uint8))
                        img_data = img_data.resize((RESIZE_DIM, RESIZE_DIM))
                        save_dir = os.path.join(preprocessed_data_dir, patient_dir)
                        os.makedirs(save_dir, exist_ok=True)
                        img_data.save(os.path.join(save_dir, f"{slice_index:04d}.png"))
                except Exception as e:
                    logging.error(f"{img_path} 로드 중 오류 발생: {e}")
            else:
                logging.warning(f"{patient_path}에서 모달리티 '{modality}' 파일을 찾을 수 없습니다.")

    if num_workers > 0:
        with multiprocessing.Pool(num_workers) as pool:
            pool.map(process_patient, patient_dirs)
    else:
        for patient_dir in patient_dirs:
            process_patient(patient_dir)

def extract_image(nifti_path, output_dir, positions, position_unit='slice', resize_dim=512):
    """
    NIfTI 파일에서 지정한 위치의 이미지를 추출하여 저장합니다.

    Args:
        nifti_path (str): NIfTI 파일의 경로
        output_dir (str): 추출된 이미지를 저장할 디렉토리
        positions (list of int or float): 추출할 위치들 (슬라이스 인덱스 또는 mm 단위)
        position_unit (str): 위치의 단위 ('slice' 또는 'mm')
        resize_dim (int): 이미지를 리사이즈할 크기 (정사각형 형태로 리사이즈됩니다)
    """
    os.makedirs(output_dir, exist_ok=True)

    if nifti_path.endswith('.nii') or nifti_path.endswith('.nii.gz'):
        # NIfTI 파일인 경우
        try:
            nii_img = nib.load(nifti_path)
            img = nii_img.get_fdata()
            orientation = io_orientation(nii_img.affine)
            affine = nii_img.affine
            # 이미지 정규화
            img = (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-5)
            num_slices = img.shape[2]

            # 슬라이스 위치 변환
            if position_unit == 'slice':
                slice_indices = [int(pos) for pos in positions]
            elif position_unit == 'mm':
                # z 방향 위치 계산
                slice_indices = []
                for pos_mm in positions:
                    # 슬라이스 인덱스로 변환
                    z0 = affine[2, 3]
                    dz = affine[2, 2]
                    slice_idx = int(round((pos_mm - z0) / dz))
                    slice_indices.append(slice_idx)
            else:
                logging.error("position_unit은 'slice' 또는 'mm'이어야 합니다.")
                return

            for slice_idx in slice_indices:
                if 0 <= slice_idx < num_slices:
                    img_slice = img[:, :, slice_idx]
                    # 방향 조정
                    if orientation[0, 0] == -1:
                        img_slice = np.flip(img_slice, axis=0)
                    if orientation[1, 1] == -1:
                        img_slice = np.flip(img_slice, axis=1)
                    img_data = Image.fromarray((img_slice * 255).astype(np.uint8))
                    img_data = img_data.resize((resize_dim, resize_dim))
                    output_path = os.path.join(output_dir, f'slice_{slice_idx:04d}.png')
                    img_data.save(output_path)
                    logging.info(f"슬라이스 {slice_idx}가 저장되었습니다: {output_path}")
                else:
                    logging.warning(f"슬라이스 인덱스 {slice_idx}가 유효한 범위를 벗어났습니다.")
        except Exception as e:
            logging.error(f"NIfTI 파일 로드 중 오류 발생: {e}")
    else:
        logging.error("지원하지 않는 파일 형식입니다. NIfTI(.nii, .nii.gz) 파일만 지원합니다.")

# ===========================
# 메인 실행 코드
# ===========================
if __name__ == "__main__":
    for run_setting in RUN_SETTINGS_LIST:
        MODE = run_setting['mode']
        MODEL_TYPE = run_setting['model_type']
        settings = run_setting['settings']

        if MODE == "preprocess":
            modality = settings['modality']
            preprocess_dataset(
                dataset_root=DATASET_ROOT,
                preprocessed_data_dir=PREPROCESS_SETTINGS['preprocessed_data_dir'],
                modality=modality,
                num_workers=PREPROCESS_SETTINGS['num_workers']
            )
            logging.info("데이터셋 전처리가 완료되었습니다.")
        elif MODE == "extract":
            nifti_path = settings['nifti_path']
            output_dir = settings['output_dir']
            positions = settings['positions']
            position_unit = settings['position_unit']
            modality = settings['modality']
            extract_image(
                nifti_path=nifti_path,
                output_dir=output_dir,
                positions=positions,
                position_unit=position_unit,
                resize_dim=RESIZE_DIM
            )
            logging.info("이미지 추출이 완료되었습니다.")
        elif MODE == "extract_image":
            nifti_path = settings['nifti_path']
            output_dir = settings['output_dir']
            positions = settings['positions']
            position_unit = settings['position_unit']
            resize_dim = RESIZE_DIM
            extract_image(
                nifti_path=nifti_path,
                output_dir=output_dir,
                positions=positions,
                position_unit=position_unit,
                resize_dim=resize_dim
            )
            logging.info("이미지 추출이 완료되었습니다.")
        elif MODE == "train":
            train_settings = settings
            modality = train_settings['modality']
            num_epochs = train_settings['num_epochs']
            learning_rate = train_settings['learning_rate']
            save_every = train_settings['save_every']
            batch_size = train_settings['batch_size']
            use_tqdm = train_settings['use_tqdm']
            model_path = train_settings['model_path']
            validation_split = train_settings['validation_split']
            window_size = train_settings['window_size']
            augment_data = train_settings['augment_data']
            loss_type = train_settings['loss_type']
            num_workers = train_settings['num_workers']
            pin_memory = train_settings['pin_memory']
            prefetch_factor = train_settings['prefetch_factor']
            model_dir = os.path.join("models", MODEL_TYPE)
            use_amp = train_settings['use_amp']
            patch_size = train_settings['patch_size']
            use_patch_training = train_settings['use_patch_training']
            scheduler_type = train_settings['scheduler_type']
            scheduler_params = train_settings['scheduler_params']

            # 모델 이름 생성
            model_name = f"{MODEL_TYPE}_{loss_type}"

            start_epoch = 0
            # 모델 초기화
            if MODEL_TYPE == "film":
                model = FILMModel()
            elif MODEL_TYPE == "diffmorpher":
                model = DiffMorpher()
            else:
                raise ValueError("잘못된 모델 선택입니다. 'film' 또는 'diffmorpher'를 선택하세요.")

            if model_path and os.path.exists(model_path):
                checkpoint = torch.load(model_path)
                generator_state_dict = checkpoint.get('generator_state_dict', checkpoint.get('model_state_dict', None))
                if generator_state_dict:
                    # DataParallel로 저장된 경우 키 수정
                    new_state_dict = {}
                    for k, v in generator_state_dict.items():
                        if k.startswith('module.'):
                            new_state_dict[k[7:]] = v
                        else:
                            new_state_dict[k] = v
                    try:
                        model.load_state_dict(new_state_dict, strict=False)
                        start_epoch = checkpoint.get('epoch', 0)
                        logging.info(f"{start_epoch} 에포크부터 학습을 재개합니다.")
                    except Exception as e:
                        logging.error(f"체크포인트 로드 중 오류 발생: {e}")
                        logging.info("새 학습을 시작합니다.")
                else:
                    logging.info("체크포인트에 'generator_state_dict'가 없습니다. 새 학습을 시작합니다.")
            else:
                logging.info("새 학습을 시작합니다.")

            # DataLoader용 Transform 설정
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5])
            ])

            # 데이터셋 및 DataLoader 설정
            dataset = BraTSDataset(
                dataset_root=DATASET_ROOT,
                transform=transform,
                modality=modality,
                mode='mmap',
                window_size=window_size,
                augment_data=augment_data,
                preprocessed=False,
                patch_size=patch_size,
                use_patch_training=use_patch_training
            )

            # 데이터셋 분할 (학습/검증)
            total_size = len(dataset)
            val_size = int(total_size * validation_split)
            train_size = total_size - val_size
            train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

            # DataLoader 설정
            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,  # num_workers를 0으로 설정
                pin_memory=pin_memory,
                prefetch_factor=prefetch_factor,
                persistent_workers=False  # 멀티프로세싱 비활성화 시에는 False로 설정
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,  # num_workers를 0으로 설정
                pin_memory=pin_memory,
                prefetch_factor=prefetch_factor,
                persistent_workers=False  # 멀티프로세싱 비활성화 시에는 False로 설정
            )

            # 장치 설정 (다중 GPU 지원)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            # 모델 학습 시작
            train_model(
                generator=model,
                train_loader=train_loader,
                val_loader=val_loader,
                num_epochs=num_epochs,
                learning_rate=learning_rate,
                save_every=save_every,
                use_tqdm=use_tqdm,
                model_dir=model_dir,
                start_epoch=start_epoch,
                loss_type=loss_type,
                use_amp=use_amp,
                device=device,
                model_name=model_name,
                scheduler_type=scheduler_type,
                scheduler_params=scheduler_params
            )
        elif MODE == "test":
            test_settings = settings
            start_image_path = test_settings['start_image_path']
            start_z_mm = test_settings['start_z_mm']
            end_image_path = test_settings['end_image_path']
            end_z_mm = test_settings['end_z_mm']
            num_frames = test_settings['num_frames']
            original_nifti_path = test_settings['original_nifti_path']
            model_path = test_settings['model_path']
            SAVE_PATH = test_settings.get('save_path', './test_output')

            # 모델 초기화
            if MODEL_TYPE == "film":
                model = FILMModel()
            elif MODEL_TYPE == "diffmorpher":
                model = DiffMorpher()
            else:
                raise ValueError("잘못된 모델 선택입니다. 'film' 또는 'diffmorpher'를 선택하세요.")

            # 장치 설정 (다중 GPU 지원)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            if not model_path:
                logging.error("테스트 모드에서는 모델 경로를 입력해야 합니다.")
                sys.exit(1)
            try:
                checkpoint = torch.load(model_path, map_location=device)
                generator_state_dict = checkpoint.get('generator_state_dict', checkpoint.get('model_state_dict', None))
                if generator_state_dict:
                    # DataParallel로 저장된 경우 키 수정
                    new_state_dict = {}
                    for k, v in generator_state_dict.items():
                        if k.startswith('module.'):
                            new_state_dict[k[7:]] = v
                        else:
                            new_state_dict[k] = v
                    model.load_state_dict(new_state_dict, strict=False)
                else:
                    logging.error("체크포인트에 'generator_state_dict'가 없습니다.")
                    sys.exit(1)
            except Exception as e:
                logging.error(f"모델 로드 중 오류 발생: {e}")
                sys.exit(1)

            model.eval()

            # 입력 이미지 로드
            try:
                start_image = Image.open(start_image_path).convert("L")
                end_image = Image.open(end_image_path).convert("L")
            except Exception as e:
                logging.error(f"입력 이미지 로드 중 오류 발생: {e}")
                sys.exit(1)

            # 중간 프레임 생성 및 저장
            test_model(
                generator=model,
                start_image=start_image,
                end_image=end_image,
                start_z_mm=start_z_mm,
                end_z_mm=end_z_mm,
                num_frames=num_frames,
                save_image=True,
                save_nifti=True,
                image_save_path=os.path.join(SAVE_PATH, 'images'),
                nifti_save_path=os.path.join(SAVE_PATH, 'nifti'),
                original_nifti_path=original_nifti_path if original_nifti_path else None,
                image_format='png',
                device=device
            )
        else:
            logging.error("잘못된 모드 선택입니다. 'train', 'test', 'preprocess', 'extract' 또는 'extract_image'를 선택하세요.")
