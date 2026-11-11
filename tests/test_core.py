import json
import math
import pickle
import tempfile
import unittest
from pathlib import Path

import h5py
import nibabel as nib
import numpy as np
import torch

from grace_unetpp import GRACEUNetPlusPlus
from grace_unetpp.model import GravityPhysics2D, GravityMLPGate, ConfidenceGuidedRefiner
from grace_unetpp.data import make_source
from grace_unetpp.evaluation import boundary_metrics_grid
from grace_unetpp.datasets.acdc import ACDCH5Dataset
from grace_unetpp.datasets.msd import CachedMSDDataset

torch.set_num_threads(2)


class OperatorTests(unittest.TestCase):
    def test_zero_dc_and_sinusoidal_response(self):
        solver=GravityPhysics2D()
        torch.testing.assert_close(solver(torch.ones(2,1,8,8)),torch.zeros(2,1,8,8))
        x=torch.arange(8,dtype=torch.float32)
        wave=torch.cos(2*math.pi*x/8).view(1,1,1,8).expand(1,1,8,8)
        expected=wave*(-1/(4*math.pi**2*(1/8)**2))
        torch.testing.assert_close(solver(wave),expected,rtol=2e-5,atol=2e-6)
        self.assertLess(abs(solver(wave).mean().item()),1e-6)

    def test_gfm_initial_identity_and_gradients(self):
        source=type('Source',(),{'GravityPhysics2D':GravityPhysics2D})
        gate=GravityMLPGate(source,4)
        x=torch.randn(2,4,8,8,requires_grad=True)
        torch.testing.assert_close(gate(x),x)
        gate.gamma.data.fill_(0.2)
        gate(x).square().mean().backward()
        for parameter in gate.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())

    def test_zero_refinement_and_confidence_gradient(self):
        refiner=ConfidenceGuidedRefiner(16,4)
        decoded=torch.randn(2,16,8,8,requires_grad=True)
        logits=torch.randn(2,4,8,8,requires_grad=True)
        torch.testing.assert_close(refiner(decoded,logits),torch.zeros_like(logits))
        torch.nn.init.normal_(refiner.block[-1].weight,std=.1)
        correction,u=refiner.correction(decoded,logits)
        self.assertTrue(((u>=0)&(u<=.75+1e-6)).all())
        refiner(decoded,logits).square().mean().backward()
        self.assertGreater(logits.grad.abs().sum().item(),0)
        self.assertGreater(decoded.grad.abs().sum().item(),0)

    def test_boundary_metric_edge_cases(self):
        empty=np.zeros((8,8),bool)
        mask=empty.copy();mask[2:6,2:6]=True
        self.assertEqual(boundary_metrics_grid(mask,mask),(0.,0.))
        self.assertTrue(all(np.isnan(v) for v in boundary_metrics_grid(empty,empty)))
        expected=np.hypot(7*2,7*3)
        hd95,asd=boundary_metrics_grid(empty,mask,(2,3))
        self.assertAlmostEqual(hd95,expected)
        self.assertAlmostEqual(asd,expected)


class ModelTests(unittest.TestCase):
    def test_acdc_and_msd_shapes_and_stage_parameters(self):
        for channels,classes in [(1,4),(3,2)]:
            with self.subTest(channels=channels):
                model=GRACEUNetPlusPlus(channels,classes,None).eval()
                with torch.no_grad():
                    refined,base,correction,u=model.forward_with_diagnostics(
                        torch.randn(1,channels,256,256))
                self.assertEqual(tuple(refined.shape),(1,classes,256,256))
                torch.testing.assert_close(refined,base)
                self.assertEqual(tuple(u.shape),(1,1,256,256))
                model.set_stage('gfm')
                self.assertTrue(all(not p.requires_grad for p in model.refiner.parameters()))
                self.assertTrue(all(p.requires_grad for p in model.gravity.parameters()))
                model.set_stage('tail')
                self.assertTrue(all(not p.requires_grad for p in model.base_model.encoder.parameters()))
                self.assertTrue(all(not p.requires_grad for p in model.gravity.parameters()))
                self.assertTrue(all(p.requires_grad for p in model.base_model.decoder.parameters()))
                self.assertTrue(all(p.requires_grad for p in model.refiner.parameters()))

    def test_checkpoint_state_roundtrip(self):
        model=GRACEUNetPlusPlus(1,4,None).eval()
        other=GRACEUNetPlusPlus(1,4,None).eval()
        other.load_state_dict(model.state_dict(),strict=True)
        x=torch.randn(1,1,64,64)
        with torch.no_grad():torch.testing.assert_close(model(x),other(x),rtol=0,atol=0)


class DataAndSplitTests(unittest.TestCase):
    def test_recorded_subject_partitions(self):
        root=Path(__file__).resolve().parents[1]
        for dataset,count in [('acdc',100),('msd',41)]:
            manifest=json.loads((root/'splits'/f'{dataset}_5fold.json').read_text())
            ids=set(manifest['burdens'])
            self.assertEqual(len(ids),count)
            tested=[]
            for row in manifest['fold_assignments']:
                train,val,test=map(set,(row['train_ids'],row['validation_ids'],row['test_ids']))
                self.assertFalse(train&val or train&test or val&test)
                self.assertEqual(train|val|test,ids)
                tested.extend(row['test_ids'])
            self.assertEqual(len(tested),count)
            self.assertEqual(set(tested),ids)

    def test_hdf5_range_labels_and_spacing(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'patient001_frame01_slice_0.h5'
            with h5py.File(path,'w') as f:
                f['image']=np.ones((32,32),np.float32)*.5
                f['label']=np.ones((32,32),np.uint8)
                f['spacing']=np.array([1.2,1.5],np.float32)
            image,target,spacing=ACDCH5Dataset([str(path)],is_train=False,return_spacing=True)[0]
            self.assertEqual(tuple(image.shape),(1,256,256))
            torch.testing.assert_close(image,torch.zeros_like(image))
            self.assertEqual(set(target.unique().tolist()),{1})
            torch.testing.assert_close(spacing,torch.tensor([1.2,1.5]))

    def test_msd_sampling_cache_and_pickle(self):
        with tempfile.TemporaryDirectory() as folder:
            folder=Path(folder)
            img=folder/'spleen_1.nii.gz';mask=folder/'spleen_1_label.nii.gz'
            volume=np.ones((20,24,3),np.float32)*40
            labels=np.zeros_like(volume,dtype=np.uint8);labels[5:10,5:10,0]=1
            nib.save(nib.Nifti1Image(volume,np.diag([1.2,1.5,2,1])),img)
            nib.save(nib.Nifti1Image(labels,np.diag([1.2,1.5,2,1])),mask)
            train=CachedMSDDataset([(str(img),str(mask))],is_train=True,sampling_seed=731)
            test=CachedMSDDataset([(str(img),str(mask))],is_train=False,return_spacing=True)
            self.assertEqual(len(train),2)
            self.assertEqual(len(test),3)
            image,target,spacing=test[0]
            self.assertEqual(tuple(image.shape),(3,256,256))
            torch.testing.assert_close(image,torch.zeros_like(image),atol=1e-6,rtol=0)
            self.assertEqual(tuple(target.shape),(256,256))
            self.assertEqual(len(pickle.loads(pickle.dumps(test))),3)


if __name__=='__main__':unittest.main()
