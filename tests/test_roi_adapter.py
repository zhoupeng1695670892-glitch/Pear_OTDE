import io
import unittest
import torch
from torch import nn
from pear_innovations.detector_bridge import DetectorBridgeConfig, DetectorBridge
from pear_innovations.roi import nms, reframe_gt_masks, sample_rois

class ToyConv(nn.Module):

    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, 3, stride=stride, padding=1)

    def forward(self, x):
        return self.conv(x).relu()

class ToyC3Interface(nn.Module):

    def __init__(self, cin, cout):
        super().__init__()
        self.cv1 = ToyConv(cin, cout)
        self.cv2 = ToyConv(cout, cout)

    def forward(self, x):
        return self.cv2(self.cv1(x))

class ToyConcat(nn.Module):

    def forward(self, x):
        return torch.cat(x, 1)

class ToyFixedDetectionDecoder(nn.Module):

    def __init__(self):
        super().__init__()
        self.nc, self.end2end, self.xyxy = (1, False, True)
        self.one2many = {}
        self.scores = nn.Parameter(torch.tensor([2.0, 1.0, 3.0]))
        self.register_buffer('predicted_boxes', torch.tensor([[8.0, 8.0, 24.0, 24.0], [9.0, 9.0, 25.0, 25.0], [40.0, 40.0, 56.0, 56.0]]))

    def forward_head(self, features, **kwargs):
        return {'scores': self.scores[None, None].expand(features[0].shape[0], 1, -1)}

    def _get_decode_boxes(self, raw):
        boxes = self.predicted_boxes
        if not self.xyxy:
            boxes = torch.cat(((boxes[:, :2] + boxes[:, 2:]) / 2, boxes[:, 2:] - boxes[:, :2]), -1)
        return boxes.T[None].expand(raw['scores'].shape[0], -1, -1)

class Toy25NodeContractGraph(nn.Module):

    def __init__(self):
        super().__init__()
        layers = [ToyConv(3, 4, 2), ToyConv(4, 4, 2), ToyC3Interface(4, 8), ToyConv(8, 8, 2), ToyC3Interface(8, 8), nn.Identity(), ToyConv(8, 8, 2), ToyC3Interface(8, 8), nn.Identity(), ToyConv(8, 16, 2), ToyC3Interface(16, 16), nn.Identity(), nn.Upsample(scale_factor=2, mode='nearest'), ToyConcat(), ToyC3Interface(24, 8), nn.Upsample(scale_factor=2, mode='nearest'), ToyConcat(), ToyC3Interface(16, 8), ToyConv(8, 8, 2), ToyConcat(), ToyC3Interface(16, 8), ToyConv(8, 8, 2), ToyConcat(), ToyC3Interface(24, 16), ToyFixedDetectionDecoder()]
        sources = [-1] * 25
        for index, source in {13: [-1, 8], 16: [-1, 5], 19: [-1, 14], 22: [-1, 11], 24: [17, 20, 23]}.items():
            sources[index] = source
        for layer, source in zip(layers, sources):
            layer.f = source
        self.model = nn.ModuleList(layers)
        self.register_buffer('stride', torch.tensor([8.0, 16.0, 32.0]))

class ROITests(unittest.TestCase):

    def test_full_image_identity_and_known_crop(self):
        feature = torch.arange(16.0).view(1, 1, 4, 4)
        full = sample_rois(feature, torch.tensor([[0.0, 0.0, 0.0, 8.0, 8.0]]), (8, 8), 4)
        torch.testing.assert_close(full, feature)
        crop = sample_rois(feature, torch.tensor([[0.0, 1.0, 1.0, 3.0, 3.0]]), (4, 4), 2)
        torch.testing.assert_close(crop, torch.tensor([[[[5.0, 6.0], [9.0, 10.0]]]]))

    def test_roi_and_feature_gradients(self):
        feature = torch.arange(16.0, dtype=torch.double).reshape(1, 1, 4, 4).requires_grad_()
        rois = torch.tensor([[0.0, 0.8, 0.9, 3.1, 3.2]], dtype=torch.double, requires_grad=True)
        fn = lambda f, r: sample_rois(f, r, (4, 4), 2)
        fn_coords = lambda coords: fn(feature, torch.cat((coords.new_zeros((1, 1)), coords), -1))
        self.assertTrue(torch.autograd.gradcheck(fn_coords, (rois[:, 1:].detach().requires_grad_(),)))
        crop = fn(feature, rois)
        crop.sum().backward()
        self.assertGreater(feature.grad.abs().sum().item(), 0)
        self.assertGreater(rois.grad[:, 1:].abs().sum().item(), 0)
        self.assertEqual(rois.grad[:, 0].item(), 0)

    def test_empty_rois_and_invalid_shapes(self):
        feature = torch.randn(2, 4, 3, 5, requires_grad=True)
        out = sample_rois(feature, torch.empty(0, 5), (24, 40), 7)
        self.assertEqual(out.shape, (0, 4, 7, 7))
        out.sum().backward()
        self.assertEqual(feature.grad.abs().sum().item(), 0)
        for roi in (torch.tensor([[0.5, 0.0, 0.0, 1.0, 1.0]]), torch.tensor([[2.0, 0.0, 0.0, 1.0, 1.0]]), torch.tensor([[0.0, 1.0, 1.0, 0.0, 2.0]])):
            with self.assertRaises(ValueError):
                sample_rois(feature, roi, (24, 40))
        with self.assertRaises(ValueError):
            sample_rois(torch.ones(1, 1, 2, 2, dtype=torch.bool), torch.empty(0, 5), (2, 2))

    def test_bool_mask_reframe_preserves_coordinates_and_soft_values(self):
        mask = torch.tensor([[[[True, False], [False, False]]]])
        boxes = torch.tensor([[0.0, 0.0, 2.0, 2.0]])
        identity = reframe_gt_masks(mask, boxes, boxes, 2)
        self.assertEqual(identity.dtype, torch.float32)
        torch.testing.assert_close(identity, mask.float())
        center = reframe_gt_masks(mask, boxes, torch.tensor([[0.5, 0.5, 1.5, 1.5]]), 1)
        torch.testing.assert_close(center, torch.tensor([[[[0.25]]]]))
        for dtype in (torch.uint8, torch.int64):
            torch.testing.assert_close(reframe_gt_masks(mask.to(dtype), boxes, boxes, 2), mask.float())

    def test_mask_reframe_validates_empty_and_detaches_gt_geometry(self):
        empty = reframe_gt_masks(torch.empty(0, 2, 3, 3, dtype=torch.bool), torch.empty(0, 4), torch.empty(0, 4), 4)
        self.assertEqual(empty.shape, (0, 2, 4, 4))
        with self.assertRaises(ValueError):
            reframe_gt_masks(torch.empty(0, 2, 3, 3), torch.empty(0, 4), torch.empty(0, 4), 0)
        gt = torch.tensor([[0.0, 0.0, 2.0, 2.0]], requires_grad=True)
        roi = torch.tensor([[0.2, 0.2, 1.8, 1.8]], requires_grad=True)
        mask = torch.arange(4.0).reshape(1, 1, 2, 2)
        reframe_gt_masks(mask, gt, roi, 2).sum().backward()
        self.assertIsNone(gt.grad)
        self.assertTrue(torch.isfinite(roi.grad).all())

    def test_nms_stable_ties_zero_and_validation(self):
        boxes = torch.tensor([[0.0, 0.0, 2.0, 2.0], [0.1, 0.1, 2.1, 2.1], [5.0, 5.0, 7.0, 7.0]])
        torch.testing.assert_close(nms(boxes, torch.ones(3), 0.5), torch.tensor([0, 2]))
        self.assertEqual(nms(torch.empty(0, 4), torch.empty(0)).numel(), 0)
        with self.assertRaises(ValueError):
            nms(boxes, torch.ones(3, 1))
        with self.assertRaises(ValueError):
            nms(boxes, torch.tensor([1.0, float('nan'), 2.0]))

class ToyAdapterContractTests(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(18)
        self.config = DetectorBridgeConfig(roi_size=5, grade_dim=8, phase_rank=4, max_detections=10)
        self.model = DetectorBridge(Toy25NodeContractGraph(), self.config)
        self.images = torch.randn(1, 3, 64, 64)

    def test_shared_original_main_and_no_new_visibility_supervision(self):
        self.assertIs(self.model.downsample.main, self.model.master.model[3])
        self.assertEqual(len(list(self.model.parameters())), len({id(p) for p in self.model.parameters()}))
        self.assertFalse(self.model.fusion.use_fine_support)
        baseline = DetectorBridge(Toy25NodeContractGraph(), DetectorBridgeConfig(lecf=False, cseg=False, rcpm=False, roi_size=5, grade_dim=8, phase_rank=4))
        self.assertIsNotNone(baseline.downsample.support_head)
        self.assertIsNotNone(baseline.grade_head.evidence_heads)

    def test_real_three_modules_joint_grade_backward(self):
        self.model.train()
        rois = torch.tensor([[0.0, 8.4, 9.1, 30.3, 31.6]], requires_grad=True)
        images = self.images.clone().requires_grad_()
        out = self.model(images, rois)
        out['grading']['grade_logits'].square().sum().backward()
        self.assertGreater(images.grad.abs().sum().item(), 0)
        self.assertGreater(rois.grad[:, 1:].abs().sum().item(), 0)
        for parameter in (self.model.master.model[0].conv.weight, self.model.downsample.projection.weight, self.model.fusion.unique_value[0].weight, self.model.grade_head.classifier.weight):
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(parameter.grad.abs().sum().item(), 0)
        self.assertIsNone(self.model.grade_head.evidence_heads.visible.weight.grad)

    def test_predict_refuses_training_mode(self):
        with self.assertRaisesRegex(RuntimeError, 'eval'):
            self.model.predict(self.images)

    def test_predict_uses_only_decoder_boxes_and_predicted_evidence(self):
        self.model.eval()
        observed = []
        hook = self.model.grade_head.register_forward_hook(lambda module, args, output: observed.append(output['aux']['evidence_override_used']))
        try:
            result = self.model.predict(self.images)[0]
        finally:
            hook.remove()
        expected_boxes = self.model.detection_head.predicted_boxes[[2, 0]]
        torch.testing.assert_close(result['boxes'], expected_boxes)
        torch.testing.assert_close(result['grade_probabilities'].sum(-1), torch.ones(2))
        torch.testing.assert_close(result['grade_scores'], result['fruit_scores'][:, None] * result['grade_probabilities'])
        self.assertEqual(observed, [False])
        self.assertFalse(result['grade_scores'].requires_grad)

    def test_xywh_decoder_convention_and_zero_detections(self):
        self.model.eval()
        xyxy_result = self.model.predict(self.images)[0]
        self.model.detection_head.xyxy = False
        xywh_result = self.model.predict(self.images)[0]
        torch.testing.assert_close(xyxy_result['boxes'], xywh_result['boxes'])
        with torch.no_grad():
            self.model.detection_head.scores.fill_(-100)
        result = self.model.predict(self.images)[0]
        self.assertEqual(result['boxes'].shape, (0, 4))
        self.assertEqual(result['grade_probabilities'].shape, (0, 3))
        self.assertEqual(result['grades'].shape, (0,))

    def test_mixed_empty_images_preserve_batch_partition(self):
        self.model.eval()

        def mixed_scores(features, **kwargs):
            return {'scores': features[0].new_tensor([[[-100.0, -100.0, -100.0]], [[2.0, 1.0, 3.0]]])}
        self.model.detection_head.forward_head = mixed_scores
        result = self.model.predict(self.images.expand(2, -1, -1, -1))
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]['grade_scores'].shape, (0, 3))
        self.assertEqual(result[1]['grade_scores'].shape, (2, 3))
        torch.testing.assert_close(result[1]['boxes'], self.model.detection_head.predicted_boxes[[2, 0]])

    def test_state_dict_roundtrip_preserves_shared_main_and_outputs(self):
        self.model.eval()
        before = self.model.predict(self.images)[0]
        buffer = io.BytesIO()
        torch.save(self.model.state_dict(), buffer)
        buffer.seek(0)
        restored = DetectorBridge(Toy25NodeContractGraph(), self.config).eval()
        restored.load_state_dict(torch.load(buffer, weights_only=True))
        self.assertIs(restored.downsample.main, restored.master.model[3])
        after = restored.predict(self.images)[0]
        for name in before:
            torch.testing.assert_close(before[name], after[name])

    def test_schema_and_input_guards(self):
        invalid = Toy25NodeContractGraph()
        invalid.model[-1].nc = 3
        with self.assertRaises(ValueError):
            DetectorBridge(invalid, self.config)
        invalid = Toy25NodeContractGraph()
        invalid.model[16].f = [-1, 4]
        with self.assertRaises(ValueError):
            DetectorBridge(invalid, self.config)
        with self.assertRaises(ValueError):
            self.model(torch.randn(1, 3, 63, 64))
if __name__ == '__main__':
    unittest.main()
