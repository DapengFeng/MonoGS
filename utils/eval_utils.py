import json
import os

import cv2
import evo
import numpy as np
import torch
from evo.core import metrics, trajectory
from evo.core.trajectory import PosePath3D
from evo.tools.settings import SETTINGS
from matplotlib import pyplot as plt
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.image.psnr import PeakSignalNoiseRatio
from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure
import torchvision

import wandb
from gaussian_splatting.gaussian_renderer import render
from gaussian_splatting.utils.image_utils import psnr
from gaussian_splatting.utils.loss_utils import ssim
from gaussian_splatting.utils.system_utils import mkdir_p
from utils.logging_utils import Log
from copy import deepcopy


def evaluate_evo(poses_gt, poses_est, plot_dir, label, monocular=False):
    ## Plot
    traj_ref = PosePath3D(poses_se3=poses_gt)
    traj_est = PosePath3D(poses_se3=poses_est)
    traj_est.align(traj_ref, correct_scale=monocular)
    traj_est_aligned = traj_est

    ## RMSE
    pose_relation = metrics.PoseRelation.translation_part
    data = (traj_ref, traj_est_aligned)
    ape_metric = metrics.APE(pose_relation)
    ape_metric.process_data(data)
    ape_stat = ape_metric.get_statistic(metrics.StatisticsType.rmse)
    ape_stats = ape_metric.get_all_statistics()
    Log("RMSE ATE \[m]", ape_stat, tag="Eval")

    with open(
        os.path.join(plot_dir, "stats_{}.json".format(str(label))),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(ape_stats, f, indent=4)

    # plot_mode = evo.tools.plot.PlotMode.xy
    # fig = plt.figure()
    # ax = evo.tools.plot.prepare_axis(fig, plot_mode)
    # ax.set_title(f"ATE RMSE: {ape_stat}")
    # evo.tools.plot.traj(ax, plot_mode, traj_ref, "--", "gray", "gt")
    # evo.tools.plot.traj_colormap(
    #     ax,
    #     traj_est_aligned,
    #     ape_metric.error,
    #     plot_mode,
    #     min_map=ape_stats["min"],
    #     max_map=ape_stats["max"],
    # )
    # ax.legend()
    # plt.savefig(os.path.join(plot_dir, "evo_2dplot_{}.png".format(str(label))), dpi=90)

    return ape_stat


def eval_ate(frames, kf_ids, save_dir, iterations, final=False, monocular=False):
    trj_data = dict()
    latest_frame_idx = kf_ids[-1] + 2 if final else kf_ids[-1] + 1
    trj_id, trj_est, trj_gt = [], [], []
    trj_est_np, trj_gt_np = [], []

    def gen_pose_matrix(R, T):
        pose = np.eye(4)
        pose[0:3, 0:3] = R.cpu().numpy()
        pose[0:3, 3] = T.cpu().numpy()
        return pose

    # for kf_id in kf_ids:
    #     kf = frames[kf_id]
    #     pose_est = np.linalg.inv(gen_pose_matrix(kf.R, kf.T))
    #     pose_gt = np.linalg.inv(gen_pose_matrix(kf.R_gt, kf.T_gt))

    #     trj_id.append(frames[kf_id].uid)
    #     trj_est.append(pose_est.tolist())
    #     trj_gt.append(pose_gt.tolist())

    #     trj_est_np.append(pose_est)
    #     trj_gt_np.append(pose_gt)
    for _, frame in frames.items():
        pose_est = np.linalg.inv(gen_pose_matrix(frame.R, frame.T))
        pose_gt = np.linalg.inv(gen_pose_matrix(frame.R_gt, frame.T_gt))

        trj_id.append(frame.uid)
        trj_est.append(pose_est.tolist())
        trj_gt.append(pose_gt.tolist())

        trj_est_np.append(pose_est)
        trj_gt_np.append(pose_gt)

    trj_data["trj_id"] = trj_id
    trj_data["trj_est"] = trj_est
    trj_data["trj_gt"] = trj_gt

    plot_dir = os.path.join(save_dir, "plot")
    mkdir_p(plot_dir)

    label_evo = "final" if final else "{:04}".format(iterations)
    with open(
        os.path.join(plot_dir, f"trj_{label_evo}.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(trj_data, f, indent=4)

    ate = evaluate_evo(
        poses_gt=trj_gt_np,
        poses_est=trj_est_np,
        plot_dir=plot_dir,
        label=label_evo,
        monocular=monocular,
    )
    wandb.log({"frame_idx": latest_frame_idx, "ate": ate})
    return ate


def eval_rendering(
    frames,
    gaussians,
    dataset,
    save_dir,
    pipe,
    background,
    kf_indices,
    iteration="final",
):
    interval = 1
    img_pred, img_gt, saved_frame_idx = [], [], []
    end_idx = len(frames)
    psnr_array, ssim_array, lpips_array = [], [], []
    psnr_mask_array, ssim_mask_array, lpips_mask_array = [], [], []
    cal_lpips = LearnedPerceptualImagePatchSimilarity(
        net_type="alex", normalize=True
    ).to("cuda")
    cal_psnr = PeakSignalNoiseRatio().to("cuda")
    cal_ssim = StructuralSimilarityIndexMeasure().to("cuda")
    render_dir = os.path.join(save_dir, "render")
    os.makedirs(render_dir, exist_ok=True)

    trj_est_np, trj_gt_np = [], []

    def gen_pose_matrix(R, T):
        pose = np.eye(4)
        pose[0:3, 0:3] = R.cpu().numpy()
        pose[0:3, 3] = T.cpu().numpy()
        return pose

    for _, frame in frames.items():
        pose_est = gen_pose_matrix(frame.R, frame.T)
        pose_gt = gen_pose_matrix(frame.R_gt, frame.T_gt)

        trj_est_np.append(pose_est)
        trj_gt_np.append(pose_gt)

    trj_est_evo = trajectory.PosePath3D(poses_se3=trj_est_np)
    trj_gt_evo = trajectory.PosePath3D(poses_se3=trj_gt_np)

    trj_gt_evo.align(trj_est_evo)

    trj_gt = torch.from_numpy(np.array(trj_gt_evo.poses_se3))

    for idx in range(0, end_idx, interval):
        # if idx in kf_indices:
        #     continue
        saved_frame_idx.append(idx)
        frame = deepcopy(frames[idx])
        gt_image, depth, _ = dataset[idx]
        gt_image = gt_image.unsqueeze(0)
        frame.update_RT(trj_gt[idx][:3, :3], trj_gt[idx][:3, 3])

        rendering = render(frame, gaussians, pipe, background)["render"]
        image = torch.clamp(rendering, 0.0, 1.0).unsqueeze(0)

        psnr_score = cal_psnr(image, gt_image)
        ssim_score = cal_ssim(image, gt_image)
        lpips_score = cal_lpips(image, gt_image)

        psnr_array.append(psnr_score.item())
        ssim_array.append(ssim_score.item())
        lpips_array.append(lpips_score.item())

        mask = gt_image.sum(1, keepdim=True) > 0

        image = image * mask
        # gt_image = gt_image * mask

        psnr_score = cal_psnr(image, gt_image)
        ssim_score = cal_ssim(image, gt_image)
        lpips_score = cal_lpips(image, gt_image)

        psnr_mask_array.append(psnr_score.item())
        ssim_mask_array.append(ssim_score.item())
        lpips_mask_array.append(lpips_score.item())

        torchvision.utils.save_image(image, os.path.join(render_dir, f"{idx:04d}.png"))

    output = dict()
    output["mean_psnr"] = float(np.mean(psnr_array))
    output["mean_psnr_mask"] = float(np.mean(psnr_mask_array))
    output["mean_ssim"] = float(np.mean(ssim_array))
    output["mean_ssim_mask"] = float(np.mean(ssim_mask_array))
    output["mean_lpips"] = float(np.mean(lpips_array))
    output["mean_lpips_mask"] = float(np.mean(lpips_mask_array))

    Log(
        f'mean psnr: {output["mean_psnr"]}, ssim: {output["mean_ssim"]}, lpips: {output["mean_lpips"]}\n',
        f'mean psnr_mask: {output["mean_psnr_mask"]}, ssim_mask: {output["mean_ssim_mask"]}, lpips_mask: {output["mean_lpips_mask"]}\n',
        f'points: {gaussians.get_xyz.shape[0]}',
        tag="Eval",
    )

    psnr_save_dir = os.path.join(save_dir, "psnr", str(iteration))
    mkdir_p(psnr_save_dir)

    json.dump(
        output,
        open(os.path.join(psnr_save_dir, "final_result.json"), "w", encoding="utf-8"),
        indent=4,
    )
    return output


def save_gaussians(gaussians, name, iteration, final=False):
    if name is None:
        return
    if final:
        point_cloud_path = os.path.join(name, "point_cloud/final")
    else:
        point_cloud_path = os.path.join(
            name, "point_cloud/iteration_{}".format(str(iteration))
        )
    gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
