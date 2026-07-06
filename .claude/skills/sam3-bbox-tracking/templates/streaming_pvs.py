"""Streaming + sliding-window memory eviction for SAM3 PVS tracking.

The stock `Sam3TrackerPredictor.init_state` preloads every video frame into one
tensor (`state["images"]`) — fine for short clips, OOMs at ~18k+ frames. This
subclass adds streaming methods (no instance state) so an existing predictor
can be adopted via ``predictor.__class__ = StreamingSam3Tracker`` and fed one
frame at a time, with old non-conditioning memory evicted by a sliding window.

The per-frame tracker step is strictly causal: it reads only the conditioning
frame(s) plus a recent window of non-conditioning memory (up to
``max(num_maskmem=7, max_obj_ptrs_in_encoder=16)`` frames back), so pruning
non-conditioning entries older than ``keep`` (default 32, 2x margin) is safe.
Every heavy method (forward_image, _get_image_feature,
_run_single_frame_inference, track_step, the memory encoder,
_add_output_per_object, _get_orig_video_res_output, add_new_points_or_box,
propagate_in_video_preflight) is reused unchanged.

Build the model with ``apply_temporal_disambiguation=False`` — otherwise
``use_memory_selection=True`` activates a longer-window selection that may
read frames older than the eviction cutoff. Adopt and use:

    sam3_model = build_sam3_video_model(apply_temporal_disambiguation=False)
    predictor = sam3_model.tracker
    predictor.__class__ = StreamingSam3Tracker
    predictor.backbone = sam3_model.detector.backbone   # required for forward_image
    state = predictor.init_stream_state(H, W)
    img0 = predictor.preprocess_frame(rgb0)
    predictor.seed_features(state, 0, img0)
    predictor.add_new_points_or_box(state, frame_idx=0, obj_id=1, box=rel_box)
    predictor.propagate_in_video_preflight(state)
    # then per frame f >= 1:
    #   img = predictor.preprocess_frame(rgb)
    #   _, _, _, video_res, scores = predictor.stream_step(state, f, img)
    #   predictor.evict(state, f, keep=32)
"""
import cv2
import numpy as np
import torch

from sam3.model.sam3_tracking_predictor import Sam3TrackerPredictor


class StreamingSam3Tracker(Sam3TrackerPredictor):
    """Adds streaming + eviction to the PVS tracker. No instance state — an
    existing `Sam3TrackerPredictor` instance can be adopted via
    `obj.__class__ = StreamingSam3Tracker`.
    """

    def init_stream_state(self, video_height: int, video_width: int, num_frames: int = 10**9):
        """Build an inference_state with no preloaded frames (init_state's
        ``video_path=None`` branch). ``num_frames`` is a large sentinel — only
        used in causal future-index guards (``t >= num_frames``) and
        ``min(num_frames, max_obj_ptrs)``, both safe with a huge value."""
        return self.init_state(
            video_height=video_height,
            video_width=video_width,
            num_frames=num_frames,
        )

    @torch.inference_mode()
    def preprocess_frame(self, rgb: np.ndarray) -> torch.Tensor:
        """RGB uint8 (H, W, 3) -> normalized (1, 3, S, S) cuda tensor matching
        the official loader: square resize to ``image_size``, /255, then
        ``(x - 0.5) / 0.5`` per channel."""
        img = cv2.resize(
            rgb,
            (self.image_size, self.image_size),
            interpolation=cv2.INTER_LINEAR,
        )
        t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        t = (t - 0.5) / 0.5
        return t.unsqueeze(0).to(self.device)

    @torch.inference_mode()
    def seed_features(self, state: dict, frame_idx: int, image: torch.Tensor):
        """Cache one frame's backbone features so the predictor's internal
        ``_get_image_feature`` hits the cache (line ~1015 of
        sam3_tracking_predictor.py) instead of indexing the absent preloaded
        ``state["images"]`` tensor. Mirrors the cache-miss body of
        ``_get_image_feature`` exactly."""
        backbone_out = self.forward_image(image)
        state["cached_features"] = {frame_idx: (image, backbone_out)}

    @torch.inference_mode()
    def stream_step(self, state: dict, frame_idx: int, image: torch.Tensor):
        """Track one new (preprocessed) frame. Replicates the per-frame body
        of ``propagate_in_video`` (sam3_tracking_predictor.py:846-873) for a
        single streamed frame. Returns
        ``(frame_idx, obj_ids, low_res_masks, video_res_masks, obj_score_logits)``.
        """
        self.seed_features(state, frame_idx, image)
        output_dict = state["output_dict"]
        current_out, pred_masks = self._run_single_frame_inference(
            inference_state=state,
            output_dict=output_dict,
            frame_idx=frame_idx,
            batch_size=self._get_obj_num(state),
            is_init_cond_frame=False,
            point_inputs=None,
            mask_inputs=None,
            reverse=False,
            run_mem_encoder=True,
        )
        output_dict["non_cond_frame_outputs"][frame_idx] = current_out
        self._add_output_per_object(state, frame_idx, current_out, "non_cond_frame_outputs")
        state["frames_already_tracked"][frame_idx] = {"reverse": False}
        low_res_masks, video_res_masks = self._get_orig_video_res_output(state, pred_masks)
        return (
            frame_idx,
            state["obj_ids"],
            low_res_masks,
            video_res_masks,
            current_out["object_score_logits"],
        )

    def evict(self, state: dict, current_frame: int, keep: int = 32):
        """Delete non-conditioning memory for frames older than
        ``current_frame - keep``, from both the multi-object output_dict and
        the per-object slices (they share the maskmem tensors, so both must
        be dropped to free memory). Conditioning frames (where the box
        prompts live) are never touched."""
        cutoff = current_frame - keep
        if cutoff <= 0:
            return

        def prune(d: dict):
            for f in [k for k in d if k < cutoff]:
                del d[f]

        prune(state["output_dict"]["non_cond_frame_outputs"])
        for obj in state["output_dict_per_obj"].values():
            prune(obj["non_cond_frame_outputs"])
        prune(state["frames_already_tracked"])
