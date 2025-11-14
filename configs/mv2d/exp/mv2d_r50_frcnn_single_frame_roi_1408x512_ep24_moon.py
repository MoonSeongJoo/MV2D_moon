_base_ = [
    '../data/single_frame_moon.py', '../detectors/maskrcnn_r50.py'
]

point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
post_range = [-61.2, -61.2, -10.0, 61.2, 61.2, 10.0]
roi_size = 7
roi_srides = [16]

model = dict(
    type='MV2D',
    use_grid_mask=dict(
        use_h=True,
        use_w=True,
        rotate=1,
        offset=False,
        ratio_range=(0.4, 0.6),
        mode=1,
        prob=0.7,
        interv_ratio=0.8
    ),
    # NOTE:
    # the FPN in faster r-cnn starts from p2
    # we use p4 (downsample rate: 16)
    base_detector=dict(
        backbone=dict(
            with_cp=False,
            dcn=dict(type='DCNv2', deform_groups=1, fallback_on_stride=False),
            stage_with_dcn=(False, False, True, True),
            # frozen_stages=4, # Freeze all stages of the backbone
        ),
    ),
    neck=dict(
        type='FPN',
        in_channels=[256, 256, 256, 256, 256],
        out_channels=256,
        start_level=2,
        end_level=2,
        num_outs=1,
    ),
    roi_head=dict(
        type='MV2DSHead',
        pc_range=point_cloud_range,
        force_fp32=True,
        use_denoise=False,

        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=roi_size, sampling_ratio=-1),
            featmap_strides=roi_srides,
            out_channels=512, ),
        voxelizer=dict(
            type='SimpleVoxelization',
            voxel_size=[0.2, 0.2, 8], 
            point_cloud_range=[0, -40, -3, 70.4, 40, 1], 
            max_num_points=32,
            max_voxels=(16000, 40000),
            ),
        # voxelizer=dict(
        #     type='SimpleVoxelization',
        #     voxel_size=[0.3, 0.3, 8],           # voxel 크기 늘림 (ex: 0.2 -> 0.3)
        #     point_cloud_range=[0, -30, -3, 60, 30, 1],  # 범위 축소
        #     max_num_points=16,                  # voxel 당 최대 점 수 감소
        #     max_voxels=(8000, 20000),           # 최대 voxel 수 감소
        # ),
        voxelnet=dict(
            type='SimpleVoxelNet',
            init_cfg=dict(type='Pretrained', 
                checkpoint='data/weights/hv_pointpillars_secfpn_sbn-all_4x8_2x_nus-3d_20210826_225857-f19d00a3.pth'),
        ),
        # corr=dict(
        #     type='COTR',
        #     num_kp=200,
        # ),
        corr=dict(
            type='COTR',
            num_kp=200,
            # --- 기존 cotr_args의 내용을 여기에 추가 ---
            max_corrs=1000,
            dim_feedforward=1024,
            backbone='resnet50',
            hidden_dim=312,
            dilation=False,
            dropout=0.1,
            nheads=8,
            layer='layer3',
            enc_layers=6,
            dec_layers=6,
            position_embedding='lin_sine',
            load_weights_freeze=False,
            # 가중치 로딩은 mmdet3d의 표준 방식인 init_cfg를 사용합니다.
            # 가중치 파일이 있다면 아래와 같이 설정합니다.
            init_cfg=dict(
                type='Pretrained',
                checkpoint='data/weights/backbone_base_corr_rev5.0_corrected.pth' # 예시 경로
                # checkpoint=None # 가중치 로딩이 필요 없을 경우
            )
        ),
        corr_loss=dict(
            type='CorrelationCycleLoss',
            corr_weight=2.0,
            cycle_weight=1.0,
        ),
        z_estimator=dict(
            type='ZEstimator',
            enc_channels=312,
            bbox_channels=256,
            uv_dim=2,
            hidden_dim=512,
            init_cfg=dict(
                type='Pretrained',
                checkpoint='data/weights/zestimator_corrected.pth' # 예시 경로
            )
            ),
        bbox_head=dict(
            type='CrossAttentionBoxHead',
            # transformer_lidar의 init_cfg를 이곳으로 옮깁니다.
            # init_cfg=dict(
            #     type='Pretrained',
            #     checkpoint='data/weights/transformer_lidar_corrected.pth',
            #     # 'bbox_head' 내부의 'transformer_lidar' 라는 이름의 모듈에 적용하라는 의미
            #     override=dict(name='transformer_lidar') 
            # ),
            num_classes=10,
            pc_range=point_cloud_range,
            transformer=dict(
                type='MV2DTransformer',
                decoder=dict(
                    type='PETRTransformerDecoder',
                    return_intermediate=True,
                    num_layers=6,
                    transformerlayers=dict(
                        type='PETRTransformerDecoderLayer',
                        attn_cfgs=[
                            dict(
                                type='FlattenMHSelfAttention',
                                embed_dims=256,
                                num_heads=8,
                                dropout=0.1),
                            dict(
                                type='PETRMultiheadAttention',
                                embed_dims=256,
                                num_heads=8,
                                dropout=0.1),
                        ],
                        feedforward_channels=2048,
                        ffn_dropout=0.1,
                        with_cp=False,  ###use checkpoint to save memory
                        operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                         'ffn', 'norm')),
                )),
            transformer_lidar=dict(
                type='MV2DTransformer_lidar',
                # init_cfg=dict(
                #     type='Pretrained',
                #     checkpoint='data/weights/transformer_lidar_corrected.pth'
                # ),
                decoder=dict(
                    type='PETRTransformerDecoder',
                    return_intermediate=True,
                    num_layers=4,
                    transformerlayers=dict(
                        type='PETRTransformerDecoderLayer',
                        attn_cfgs=[
                            dict(
                                type='FlattenMHSelfAttention',
                                embed_dims=256,
                                num_heads=8,
                                dropout=0.1),
                            dict(
                                type='PETRMultiheadAttention',
                                embed_dims=256,
                                num_heads=8,
                                dropout=0.1),
                        ],
                        feedforward_channels=2048,
                        ffn_dropout=0.1,
                        with_cp=False,  ###use checkpoint to save memory
                        operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                         'ffn', 'norm')),
                )),
            bbox_coder=dict(
                type='NMSFreeCoder',
                post_center_range=post_range,
                pc_range=point_cloud_range,
                max_num=300,
                num_classes=10),
            code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.5, 1.5, 2.0, 2.0],
            loss_cls=dict(
                type='FocalLoss',
                use_sigmoid=True,
                gamma=2.0,
                alpha=0.25,
                loss_weight=2.0,
            ),
            loss_bbox=dict(type='L1Loss', loss_weight=0.25),
        ),
        pe=dict(
            positional_encoding=dict(
                type='SinePositionalEncoding3D', num_feats=128, normalize=True),
            strides=roi_srides,
            position_range=post_range,
            depth_num=64,
            with_fpe=True,
        ),
        box_correlation=dict(
            correlation_mode='topk_matched:1:0.0:0.0',
        ),
    ),
    train_cfg=dict(
        complement_2d_gt=0.4,
        detection_proposal=dict(
            score_thr=0.05,
            nms_pre=1000,
            max_per_img=75,
            nms=dict(type='nms', iou_threshold=0.6, class_agnostic=True, ),
            min_bbox_size=8),
        rcnn=dict(
            stage_loss_weights=[0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
            assigner=dict(
                type='HungarianAssigner3D',
                cls_cost=dict(type='FocalLossCost', weight=2.0),
                reg_cost=dict(type='BBox3DL1Cost', weight=0.25),
                iou_cost=dict(type='IoUCost', weight=0.0),
                # Fake cost. This is just to make it compatible with DETR head.
                pc_range=point_cloud_range),
            sampler_cfg=dict(type='PseudoSampler'),
            pos_weight=-1,
            debug=False)
    ),
    test_cfg=dict(
        detection_proposal=dict(
            score_thr=0.05,
            nms_pre=1000,
            max_per_img=75,
            nms=dict(type='nms', iou_threshold=0.6, class_agnostic=True, ),
            min_bbox_size=8),
        rcnn=dict(
            score_thr=0.0,
            nms=dict(nms_thr=1.0, use_rotate_nms=True, ),
            max_per_scene=300,
        ))
)

data = dict(
    workers_per_gpu=8,
)

optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=2e-4,
    paramwise_cfg=dict(
        custom_keys={
            'base_detector.backbone': dict(lr_mult=0.01),
            'roi_head.corr': dict(lr_mult=0.1),
            'roi_head.z_estimator': dict(lr_mult=0.1),
            'roi_head.lidar_voxelnet': dict(lr_mult=0.01),
        }
    ),
    weight_decay=0.01
    # weight_decay=0.1  # 10x 증가
    )

optimizer_config = dict(
    _delete_=True,
    grad_clip=dict(max_norm=35, norm_type=2)
    # 수정 제안 (검색 결과[5][6] 참조)
    # grad_clip=dict(max_norm=5.0, norm_type=2)
)

### epoch 기반  runner ######
total_epochs = 72

# 학습 재개를 위한 설정
load_from = None
# load_from = 'data/weights/epoch_48.pth' #check point path
# resume_from = 'data/work_dirs/20250822_lidar_camera_fusion/latest.pth'  # 같은 체크포인트 경로
resume_from = None
runner = dict(type='EpochBasedRunner', max_epochs=total_epochs)
evaluation = dict(interval=72, )
# evaluation = dict(interval=5, by_epoch=False, start=0) # validation 만 실행

# checkpoint_config 추가
checkpoint_config = dict(interval=1)  # 매 epoch마다 저장

# # Main switch to turn ON AMP for the entire training
# fp16 = dict(loss_scale=512.)

# # 수정된 설정 (기존 epoch 대신 iteration 기준 사용)
# checkpoint_config = dict(
#     interval=200,      # 300 iteration마다 저장
#     by_epoch=False,     # epoch 대신 iteration 기준 사용
#     save_optimizer=True # 옵티마이저 상태도 함께 저장
# )

find_unused_parameters = False
log_config = dict(interval=50)
lr_config = dict(
    _delete_=True,
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3,
)

# lr_config = dict(
#     _delete_=True,
#     policy='Step',
#     step=[5,10,15,20,30,40,50,60,70,80,90,100],  # 8번째와 16번째 에포크에서 학습률 감소
#     gamma=0.5,  # 각 스텝에서 학습률을 0.1배로 감소
#     # warmup='linear',
#     # warmup_iters=500,
#     # warmup_ratio=1.0 / 3,
# )

# lr_config = dict(
#     _delete_=True,
#     policy='Exp',  # Exponential decay
#     gamma=0.9,     # 각 step마다 lr을 0.9배씩 감소 (10% 감소)
#     warmup='linear',
#     warmup_iters=500,
#     warmup_ratio=1.0 / 3,
# )
