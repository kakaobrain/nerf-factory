# ------------------------------------------------------------------------------------
# NeRF-Factory
# Copyright (c) 2022 POSTECH, KAIST, Kakao Brain Corp. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------

import json
import os

import numpy as np
import pytorch_lightning as pl
import torch

# Full-reference metrics
from piqa.ssim import SSIM
from piqa.ssim import MS_SSIM
from piqa.gmsd import GMSD
from piqa.gmsd import MS_GMSD
from piqa.mdsi import MDSI
from piqa.haarpsi import HaarPSI
from piqa.vsi import VSI
from piqa.fsim import FSIM
# Feature-based metrics
from piqa.lpips import LPIPS
from piq.fid import FID

import utils.store_image as store_image

reshape_2d = lambda x: x.reshape((x.shape[0], -1))
clip_0_1 = lambda x: torch.clip(x, 0, 1).detach()


class LitModel(pl.LightningModule):

    # Utils to reorganize output values from evaluation steps,
    # i.e., validation and test step.
    def alter_gather_cat(self, outputs, key, image_sizes):
        each = torch.cat([output[key] for output in outputs])
        all = self.all_gather(each).detach()
        if all.dim() == 3:
            all = all.permute((1, 0, 2)).flatten(0, 1)
        ret, curr = [], 0
        for (h, w) in image_sizes:
            ret.append(all[curr : curr + h * w].reshape(h, w, 3))
            curr += h * w
        return ret

    @torch.no_grad()
    def psnr_each(self, preds, gts):
        psnr_list = []
        for (pred, gt) in zip(preds, gts):
            pred = torch.clip(pred, 0, 1)
            gt = torch.clip(gt, 0, 1)
            mse = torch.mean((pred - gt) ** 2)
            psnr = -10.0 * torch.log(mse) / np.log(10)
            psnr_list.append(psnr)
        return torch.stack(psnr_list)

    @torch.no_grad()
    def ssim_each(self, preds, gts):
        ssim_model = SSIM().to(device=self.device)
        ssim_list = []
        for (pred, gt) in zip(preds, gts):
            pred = torch.clip(pred.permute((2, 0, 1)).unsqueeze(0).float(), 0, 1)
            gt = torch.clip(gt.permute((2, 0, 1)).unsqueeze(0).float(), 0, 1)
            ssim = ssim_model(pred, gt)
            ssim_list.append(ssim)
        del ssim_model
        return torch.stack(ssim_list)

    @torch.no_grad()
    def msssim_each(self, preds, gts):
        msssim_model = MS_SSIM().to(device=self.device)
        msssim_list = []
        for (pred, gt) in zip(preds, gts):
            pred = torch.clip(pred.permute((2, 0, 1)).unsqueeze(0).float(), 0, 1)
            gt = torch.clip(gt.permute((2, 0, 1)).unsqueeze(0).float(), 0, 1)
            msssim = msssim_model(pred, gt)
            msssim_list.append(msssim)
        del msssim_model
        return torch.stack(msssim_list)

    @torch.no_grad()
    def gmsd_each(self, preds, gts):
        gmsd_model = GMSD().to(device=self.device)
        gmsd_list = []
        for (pred, gt) in zip(preds, gts):
            pred = torch.clip(pred.permute((2, 0, 1)).unsqueeze(0).float(), 0, 1)
            gt = torch.clip(gt.permute((2, 0, 1)).unsqueeze(0).float(), 0, 1)
            gmsd = gmsd_model(pred, gt)
            gmsd_list.append(gmsd)
        del gmsd_model
        return torch.stack(gmsd_list)

    @torch.no_grad()
    def msgmsd_each(self, preds, gts):
        msgmsd_model = MS_GMSD().to(device=self.device)
        msgmsd_list = []
        for (pred, gt) in zip(preds, gts):
            pred = torch.clip(pred.permute((2, 0, 1)).unsqueeze(0).float(), 0, 1)
            gt = torch.clip(gt.permute((2, 0, 1)).unsqueeze(0).float(), 0, 1)
            msgmsd = msgmsd_model(pred, gt)
            msgmsd_list.append(msgmsd)
        del msgmsd_model
        return torch.stack(msgmsd_list)

    @torch.no_grad()
    def mdsi_each(self, preds, gts):
        mdsi_model = MDSI().to(device=self.device)
        mdsi_list = []
        for (pred, gt) in zip(preds, gts):
            pred = torch.clip(pred.permute((2, 0, 1)).unsqueeze(0).float(), 0, 1)
            gt = torch.clip(gt.permute((2, 0, 1)).unsqueeze(0).float(), 0, 1)
            mdsi = mdsi_model(pred, gt)
            mdsi_list.append(mdsi)
        del mdsi_model
        return torch.stack(mdsi_list)

    @torch.no_grad()
    def haarpsi_each(self, preds, gts):
        haarpsi_model = HaarPSI().to(device=self.device)
        haarpsi_list = []
        for (pred, gt) in zip(preds, gts):
            pred = torch.clip(pred.permute((2, 0, 1)).unsqueeze(0).float(), 0, 1)
            gt = torch.clip(gt.permute((2, 0, 1)).unsqueeze(0).float(), 0, 1)
            haarpsi = haarpsi_model(pred, gt)
            haarpsi_list.append(haarpsi)
        del haarpsi_model
        return torch.stack(haarpsi_list)

    @torch.no_grad()
    def vsi_each(self, preds, gts):
        vsi_model = VSI().to(device=self.device)
        vsi_list = []
        for (pred, gt) in zip(preds, gts):
            pred = torch.clip(pred.permute((2, 0, 1)).unsqueeze(0).float(), 0, 1)
            gt = torch.clip(gt.permute((2, 0, 1)).unsqueeze(0).float(), 0, 1)
            vsi = vsi_model(pred, gt)
            vsi_list.append(vsi)
        del vsi_model
        return torch.stack(vsi_list)

    @torch.no_grad()
    def fsim_each(self, preds, gts):
        fsim_model = FSIM().to(device=self.device)
        fsim_list = []
        for (pred, gt) in zip(preds, gts):
            pred = torch.clip(pred.permute((2, 0, 1)).unsqueeze(0).float(), 0, 1)
            gt = torch.clip(gt.permute((2, 0, 1)).unsqueeze(0).float(), 0, 1)
            fsim = fsim_model(pred, gt)
            fsim_list.append(fsim)
        del fsim_model
        return torch.stack(fsim_list)

    @torch.no_grad()
    def lpips_each(self, preds, gts):
        lpips_model = LPIPS(network="vgg").to(device=self.device)
        lpips_list = []
        for (pred, gt) in zip(preds, gts):
            pred = torch.clip(pred.permute((2, 0, 1)).unsqueeze(0).float(), 0, 1)
            gt = torch.clip(gt.permute((2, 0, 1)).unsqueeze(0).float(), 0, 1)
            lpips = lpips_model(pred, gt)
            lpips_list.append(lpips)
        del lpips_model
        return torch.stack(lpips_list)

    @torch.no_grad()
    def fid_each(self, preds, gts):
        fid_model = FID().to(device=self.device)
        fid_list = []
        for (pred, gt) in zip(preds, gts):
            pred = torch.clip(pred.permute((2, 0, 1)).unsqueeze(0).float(), 0, 1).flatten(2).squeeze(0).movedim(0,-1)
            gt = torch.clip(gt.permute((2, 0, 1)).unsqueeze(0).float(), 0, 1).flatten(2).squeeze(0).movedim(0,-1)
            fid = fid_model(pred,gt)
            fid_list.append(fid)
        del fid_model
        return torch.stack(fid_list)

    @torch.no_grad()
    def psnr(self, preds, gts, i_train, i_val, i_test):
        ret = {}
        ret["name"] = "PSNR"
        psnr_list = self.psnr_each(preds, gts)
        ret["mean"] = psnr_list.mean().item()
        if self.trainer.datamodule.eval_test_only:
            ret["test"] = psnr_list.mean().item()
        else:
            ret["train"] = psnr_list[i_train].mean().item()
            ret["val"] = psnr_list[i_val].mean().item()
            ret["test"] = psnr_list[i_test].mean().item()

        return ret

    @torch.no_grad()
    def ssim(self, preds, gts, i_train, i_val, i_test):
        ret = {}
        ret["name"] = "SSIM"
        ssim_list = self.ssim_each(preds, gts)
        ret["mean"] = ssim_list.mean().item()
        if self.trainer.datamodule.eval_test_only:
            ret["test"] = ssim_list.mean().item()
        else:
            ret["train"] = ssim_list[i_train].mean().item()
            ret["val"] = ssim_list[i_val].mean().item()
            ret["test"] = ssim_list[i_test].mean().item()

        return ret

    @torch.no_grad()
    def msssim(self, preds, gts, i_train, i_val, i_test):
        ret = {}
        ret["name"] = "MS_SSIM"
        msssim_list = self.msssim_each(preds, gts)
        ret["mean"] = msssim_list.mean().item()
        if self.trainer.datamodule.eval_test_only:
            ret["test"] = msssim_list.mean().item()
        else:
            ret["train"] = msssim_list[i_train].mean().item()
            ret["val"] = msssim_list[i_val].mean().item()
            ret["test"] = msssim_list[i_test].mean().item()

        return ret

    @torch.no_grad()
    def gmsd(self, preds, gts, i_train, i_val, i_test):
        ret = {}
        ret["name"] = "GMSD"
        gmsd_list = self.gmsd_each(preds, gts)
        ret["mean"] = gmsd_list.mean().item()
        if self.trainer.datamodule.eval_test_only:
            ret["test"] = gmsd_list.mean().item()
        else:
            ret["train"] = gmsd_list[i_train].mean().item()
            ret["val"] = gmsd_list[i_val].mean().item()
            ret["test"] = gmsd_list[i_test].mean().item()

        return ret

    @torch.no_grad()
    def msgmsd(self, preds, gts, i_train, i_val, i_test):
        ret = {}
        ret["name"] = "MS_GMSD"
        msgmsd_list = self.msgmsd_each(preds, gts)
        ret["mean"] = msgmsd_list.mean().item()
        if self.trainer.datamodule.eval_test_only:
            ret["test"] = msgmsd_list.mean().item()
        else:
            ret["train"] = msgmsd_list[i_train].mean().item()
            ret["val"] = msgmsd_list[i_val].mean().item()
            ret["test"] = msgmsd_list[i_test].mean().item()

        return ret

    @torch.no_grad()
    def mdsi(self, preds, gts, i_train, i_val, i_test):
        ret = {}
        ret["name"] = "MDSI"
        mdsi_list = self.mdsi_each(preds, gts)
        ret["mean"] = mdsi_list.mean().item()
        if self.trainer.datamodule.eval_test_only:
            ret["test"] = mdsi_list.mean().item()
        else:
            ret["train"] = mdsi_list[i_train].mean().item()
            ret["val"] = mdsi_list[i_val].mean().item()
            ret["test"] = mdsi_list[i_test].mean().item()

        return ret

    @torch.no_grad()
    def haarpsi(self, preds, gts, i_train, i_val, i_test):
        ret = {}
        ret["name"] = "HaarPSI"
        haarpsi_list = self.haarpsi_each(preds, gts)
        ret["mean"] = haarpsi_list.mean().item()
        if self.trainer.datamodule.eval_test_only:
            ret["test"] = haarpsi_list.mean().item()
        else:
            ret["train"] = haarpsi_list[i_train].mean().item()
            ret["val"] = haarpsi_list[i_val].mean().item()
            ret["test"] = haarpsi_list[i_test].mean().item()

        return ret

    @torch.no_grad()
    def vsi(self, preds, gts, i_train, i_val, i_test):
        ret = {}
        ret["name"] = "VSI"
        vsi_list = self.vsi_each(preds, gts)
        ret["mean"] = vsi_list.mean().item()
        if self.trainer.datamodule.eval_test_only:
            ret["test"] = vsi_list.mean().item()
        else:
            ret["train"] = vsi_list[i_train].mean().item()
            ret["val"] = vsi_list[i_val].mean().item()
            ret["test"] = vsi_list[i_test].mean().item()

        return ret

    @torch.no_grad()
    def fsim(self, preds, gts, i_train, i_val, i_test):
        ret = {}
        ret["name"] = "FSIM"
        fsim_list = self.fsim_each(preds, gts)
        ret["mean"] = fsim_list.mean().item()
        if self.trainer.datamodule.eval_test_only:
            ret["test"] = fsim_list.mean().item()
        else:
            ret["train"] = fsim_list[i_train].mean().item()
            ret["val"] = fsim_list[i_val].mean().item()
            ret["test"] = fsim_list[i_test].mean().item()

        return ret

    @torch.no_grad()
    def lpips(self, preds, gts, i_train, i_val, i_test):
        ret = {}
        ret["name"] = "LPIPS"
        lpips_list = self.lpips_each(preds, gts)
        ret["mean"] = lpips_list.mean().item()
        if self.trainer.datamodule.eval_test_only:
            ret["test"] = lpips_list.mean().item()
        else:
            ret["train"] = lpips_list[i_train].mean().item()
            ret["val"] = lpips_list[i_val].mean().item()
            ret["test"] = lpips_list[i_test].mean().item()

        return ret

    def fid(self, preds, gts, i_train, i_val, i_test):
        ret = {}
        ret["name"] = "FID"
        fid_list = self.fid_each(preds, gts)
        ret["mean"] = fid_list.mean().item()
        if self.trainer.datamodule.eval_test_only:
            ret["test"] = fid_list.mean().item()
        else:
            ret["train"] = fid_list[i_train].mean().item()
            ret["val"] = fid_list[i_val].mean().item()
            ret["test"] = fid_list[i_test].mean().item()

        return ret

    def l2_normalize(x, eps=torch.finfo(torch.float32).eps):
        return x / torch.sqrt(
            torch.fmax(torch.sum(x**2, dim=-1, keepdims=True), torch.full_like(x, eps))
        )

    def write_stats(self, fpath, *stats):

        d = {}
        for stat in stats:
            d[stat["name"]] = {
                k: float(w)
                for (k, w) in stat.items()
                if k != "name" and k != "scene_wise"
            }

        with open(fpath, "w") as fp:
            json.dump(d, fp, indent=4, sort_keys=True)

    def predict_step(self, *args, **kwargs):
        return self.test_step(*args, **kwargs)

    def on_predict_epoch_end(self, outputs):
        dmodule = self.trainer.datamodule
        image_sizes = dmodule.image_sizes
        image_num = len(dmodule.render_poses)
        all_image_sizes = np.stack([image_sizes[0] for _ in range(image_num)])
        rgbs = self.alter_gather_cat(outputs[0], "rgb", all_image_sizes)

        if self.trainer.is_global_zero:
            image_dir = os.path.join(self.logdir, "render_video")
            os.makedirs(image_dir, exist_ok=True)
            store_image.store_image(image_dir, rgbs)
            store_image.store_video(image_dir, rgbs, None)

        return None
