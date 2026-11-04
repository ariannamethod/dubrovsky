import unittest
import numpy as np
from weight_geometry import bf16_bits, bf16_float, infer_direction, project_target, compare_target, norm_matched_control, norm


class WeightGeometryTests(unittest.TestCase):
    def test_bf16_roundtrip_and_ties_to_even(self):
        raw=np.array([0x3f800000,0x3f808000,0x3f818000,0xbf800000],dtype=np.uint32)
        self.assertEqual(bf16_bits(raw.view(np.float32)).tolist(),[0x3f80,0x3f80,0x3f82,0xbf80])
        all_finite=np.arange(0x7f80,dtype=np.uint16)
        np.testing.assert_array_equal(bf16_bits(bf16_float(all_finite)),all_finite)

    def test_rounding_overflow_and_invalid_values_fail(self):
        for value in [np.inf,np.nan,np.finfo(np.float32).max]:
            with self.assertRaises(ValueError):bf16_bits([value])
        with self.assertRaises(ValueError):bf16_float(np.array([0x7f80],dtype=np.uint16))

    def test_known_rank_one_edit(self):
        base=bf16_bits(np.arange(48).reshape(8,6)/8)
        u=np.zeros(8);u[2]=1
        edit=project_target(base,u,[1,4],1)
        inferred,stats=infer_direction(base,edit,[1,4])
        self.assertAlmostEqual(abs(inferred@u),1)
        self.assertAlmostEqual(stats['rank_one_energy_fraction'],1)
        plus=project_target(base,inferred,[1,4],1)
        np.testing.assert_array_equal(project_target(base,-inferred,[1,4],1),plus)
        np.testing.assert_allclose(bf16_float(plus),bf16_float(edit),rtol=0,atol=1e-12)

    def test_transfer_recomputes_target_coefficients(self):
        base=bf16_bits(np.ones((4,5)))
        target=bf16_bits(np.ones((4,5))*2)
        u=np.array([1.,0,0,0])
        actual=project_target(target,u,[1,3],1)
        self.assertEqual(bf16_float(actual)[0,1],0)
        np.testing.assert_array_equal(actual[:,[0,2,4]],target[:,[0,2,4]])
        self.assertNotEqual(bf16_float(actual)[0,1],1)  # adding the base delta leaves 1
        stats=compare_target(base,target,u,[1,3])
        self.assertAlmostEqual(stats['selected_coefficient_relative_change'],1)

    def test_zero_strength_and_bad_support(self):
        b=bf16_bits(np.ones((4,5)));u=np.array([1.,0,0,0])
        np.testing.assert_array_equal(project_target(b,u,[1],0),b)
        with self.assertRaises(ValueError):infer_direction(b,project_target(b,u,[1],1),[2])
        for cols in [[-1],[5],[1,1],[]]:
            with self.assertRaises(ValueError):project_target(b,u,cols,1)

    def test_orthogonal_delta(self):
        b=bf16_bits(np.ones((4,5)));t=bf16_float(b);t[1]+=1
        stats=compare_target(b,bf16_bits(t),np.array([1.,0,0,0]),[1,3])
        self.assertEqual(stats['direction_overlap_norm_fraction'],0)

    def test_random_control_matches_norm_and_preserves_support(self):
        b=bf16_bits(np.random.default_rng(4).normal(size=(64,16)))
        u=np.zeros(64);u[0]=1
        a,info=norm_matched_control(b,u,[1,3,7],3,123)
        again,_=norm_matched_control(b,u,[1,3,7],3,123)
        np.testing.assert_array_equal(a,again)
        self.assertAlmostEqual(info['direction_cosine'],0,places=12)
        self.assertLess(abs(norm(bf16_float(a)-bf16_float(b))/info['requested_pre_rounding_norm']-1),.01)
        np.testing.assert_array_equal(a[:,[0,2,4]],b[:,[0,2,4]])


if __name__=='__main__':unittest.main()
