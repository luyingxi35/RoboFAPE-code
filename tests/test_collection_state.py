import unittest
from robofape.collection_state import next_handoff,handoff_boundaries


class CollectionStateTest(unittest.TestCase):
    def test_handoffs_are_pre_stage_and_include_reset(self):
        ref={"stage":["reach","reach","lift","lift","place"]}
        self.assertEqual(handoff_boundaries(ref),{"reset":0,"lift":2,"place":4})

    def test_accepted_quotas_not_attempt_count(self):
        refs=[{"stage":["reach","lift","place"]}]
        accepted=[]
        for attempt in [0,4,8,10,11]:
            index,stage,step,quotas=next_handoff(refs,accepted,5,attempt)
            accepted.append({"handoff_stage":stage})
        self.assertEqual([x["handoff_stage"] for x in accepted],["reset","reset","lift","lift","place"])
        self.assertIsNone(next_handoff(refs,accepted,5,12))

    def test_reference_without_requested_stage_is_not_used(self):
        refs=[{"stage":["reach","lift"]},{"stage":["reach","place"]}]
        index,stage,step,_=next_handoff(refs,[{"handoff_stage":"reset"},{"handoff_stage":"lift"}],3,0)
        self.assertEqual((index,stage,step),(1,"place",1))

if __name__=="__main__":unittest.main()
