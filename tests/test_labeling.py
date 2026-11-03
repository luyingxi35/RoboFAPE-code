import unittest
import numpy as np
from robofape.labeling import label_profile, match_stage, monotone_segments, temporal_progress


def profile(identifier, distances, stages=None, success=False, task="task"):
    return {"id":identifier,"task_id":task,"success":success,"stage":stages or ["reach"]*len(distances),"distance":distances}


class LabelingTest(unittest.TestCase):
    def test_temporal_post_action_convention(self):
        np.testing.assert_equal(temporal_progress(4),[.25,.5,.75,1])
        np.testing.assert_equal(temporal_progress(1),[1])
        with self.assertRaises(ValueError): temporal_progress(0)

    def test_reference_selected_by_stage_initial_scalar(self):
        refs=[profile("a",[10,0],success=True),profile("b",[2,1],success=True)]
        out=label_profile(profile("f",[2,1]),refs)
        self.assertEqual([r["reference_id"] for r in out["alignment"]],["b","b"])
        self.assertEqual(out["progress"],[.5,1])

    def test_selects_separate_reference_each_stage(self):
        refs=[profile("a",[1,0,8,7],["a","a","b","b"],True),profile("b",[8,7,1,0],["a","a","b","b"],True)]
        out=label_profile(profile("f",[1,0,1,0],["a","a","b","b"]),refs)
        self.assertEqual([r["reference_id"] for r in out["alignment"]],["a","a","b","b"])

    def test_regression_is_preserved(self):
        ref=profile("r",[3,2,1,0],success=True)
        out=label_profile(profile("f",[2,0,3]),[ref])
        self.assertEqual(out["progress"],[.5,1,.25])

    def test_distance_filter_prevents_continuity_lock(self):
        # Last query has a unique close state in the late segment.
        out,_=match_stage([0,.2],[0,1,.2])
        self.assertEqual([m.reference_index for m in out],[0,2])

    def test_epsilon_boundary(self):
        # At 0.001, the early segment may retain continuity despite a closer late candidate.
        at,_=match_stage([0,1.001],[0,1,0,1.001])
        beyond,_=match_stage([0,1.00101],[0,1,0,1.00101])
        self.assertEqual(at[-1].reference_index,1)
        self.assertEqual(beyond[-1].reference_index,3)

    def test_ties_independent_of_reference_order(self):
        refs=[profile("z",[1,0],success=True),profile("a",[1,0],success=True)]
        out=label_profile(profile("f",[1]),refs)
        self.assertEqual(out["alignment"][0]["reference_id"],"a")

    def test_monotone_segments_cover_turns_and_plateaus(self):
        values=np.array([3,2,2,1,2,2,0],float)
        segments=monotone_segments(values)
        self.assertEqual(set(np.concatenate(segments)),set(range(len(values))))
        for indices in segments:
            delta=np.diff(values[indices]);self.assertTrue(np.all(delta>=0) or np.all(delta<=0))

    def test_constant_and_single_frame(self):
        matches,scale=match_stage([0,2],[1])
        self.assertEqual([x.reference_index for x in matches],[0,0])
        self.assertGreater(scale["sigma_d"],0)
        matches,_=match_stage([5],[5,5,5]);self.assertEqual(matches[0].reference_index,0)

    def test_stage_reentry_resets_and_selects_reference_again(self):
        ref=profile("r",[3,2,1,0],["a","a","b","b"],True)
        out=label_profile(profile("f",[2,0,3],["a","b","a"]),[ref])
        self.assertEqual(out["progress"],[.5,1,.25])

    def test_invalid_inputs(self):
        ref=profile("r",[1,0],success=True)
        for bad in [profile("f",[float("nan")]),profile("f",[1],["missing"]),profile("f",[1,0],["reach"])]:
            with self.assertRaises(ValueError): label_profile(bad,[ref])
        with self.assertRaises(ValueError):label_profile(profile("f",[0]),[ref,ref])
        with self.assertRaises(ValueError):label_profile(profile("f",[0]),[profile("r",[1],success=False)])


if __name__=="__main__": unittest.main()
